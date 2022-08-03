import copy
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertForPreTraining
from federatedscope.register import register_model
from federatedscope.nlp.loss import LabelSmoothingLoss
from federatedscope.nlp.model.decoder import TransformerDecoder


class ModelOutput(object):
    def __init__(self, loss, logits, hidden_states, attentions):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions


# Build your torch or tf model class here
class FedNLPModel(nn.Module):
    def __init__(self, config):  # server config
        super().__init__()

        self.encoder = BertForPreTraining.from_pretrained(config.bert_type)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout_prob = self.encoder.config.hidden_dropout_prob

        # For NLU
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.ModuleDict()
        all_tasks = [k for k in config.num_labels.keys() if k != 'cfg_check_funcs']
        self.all_labels = {k: config.num_labels[k] for k in all_tasks}
        for t, num_lb in self.all_labels.items():
            num_lb = config.num_labels[t]
            if num_lb is not None:
                self.classifier[t] = nn.Linear(self.hidden_size, num_lb)

        # For NLG
        self.vocab_size = self.encoder.config.vocab_size
        self.padding_idx = self.encoder.config.pad_token_id
        tgt_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.padding_idx)
        tgt_embeddings.weight = copy.deepcopy(self.encoder.bert.embeddings.word_embeddings.weight)
        self.decoder = TransformerDecoder(
            num_layers=config.num_dec_layers,
            d_model=self.hidden_size,
            heads=self.encoder.config.num_attention_heads,
            d_ff=self.encoder.config.intermediate_size,
            dropout=self.dropout_prob,
            embeddings=tgt_embeddings,
        )
        self.generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.vocab_size),
            nn.LogSoftmax(dim=-1),
        )
        self.generator[0].weight = self.decoder.embeddings.weight
        self._init_decoder_params()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        config=None,  # client specific config
    ):
        outputs = self.encoder.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        task = config.data.task
        if task == 'pretrain':
            collator = config.data.collator
            if collator == 'mlm':
                logits = self.encoder.cls.predictions(outputs.last_hidden_state)
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
                loss = masked_lm_loss

            elif collator == 'denoise':
                target_ids = labels
                dec_state = self.decoder.init_decoder_state(input_ids, outputs.last_hidden_state)
                decoder_outputs, state = self.decoder(target_ids[:, :-1], outputs.last_hidden_state, dec_state)
                logits = self.generator(decoder_outputs.view(-1, decoder_outputs.size(2)))
                loss_fct = nn.NLLLoss(
                    ignore_index=self.padding_idx,
                    reduction='sum',
                )
                num_tokens = target_ids[:, 1:].ne(self.padding_idx).sum().item()
                loss = loss_fct(logits, target_ids[:, 1:].contiguous().view(-1)) / num_tokens

        elif task in {'imdb', 'agnews'}:
            pooled_output = self.dropout(outputs.pooler_output)
            logits = self.classifier[task](pooled_output)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        elif task in {'squad', 'newsqa'}:
            logits = self.classifier[task](outputs.last_hidden_state)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
            logits = (start_logits, end_logits)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        elif task in {'cnndm', 'msqg'}:
            target_ids = labels
            dec_state = self.decoder.init_decoder_state(input_ids, outputs.last_hidden_state)
            decoder_outputs, state = self.decoder(target_ids[:, :-1], outputs.last_hidden_state, dec_state)
            logits = self.generator(decoder_outputs.view(-1, decoder_outputs.size(-1)))

            label_smoothing = config.model.label_smoothing if self.training else 0.0
            if label_smoothing > 0:
                loss_fct = LabelSmoothingLoss(
                    label_smoothing,
                    self.vocab_size,
                    ignore_index=self.padding_idx,
                ).to(logits.device)
            else:
                loss_fct = nn.NLLLoss(
                    ignore_index=self.padding_idx,
                    reduction='sum',
                )
            num_tokens = target_ids[:, 1:].ne(self.padding_idx).sum().item()
            loss = loss_fct(logits, target_ids[:, 1:].contiguous().view(-1)) / num_tokens

        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _init_decoder_params(self):
        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        for p in self.generator.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                p.data.zero_()


def call_fednlp_model(model_config, local_data):
    if model_config.type == 'fednlp_model':
        model = FedNLPModel(model_config)
        return model


register_model('fednlp_model', call_fednlp_model)
