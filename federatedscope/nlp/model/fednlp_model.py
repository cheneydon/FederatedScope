import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.encoder_decoder import EncoderDecoderModel
from federatedscope.register import register_model
from federatedscope.nlp.loss import LabelSmoothingLoss


class ModelOutput(object):
    def __init__(self, loss=None, logits=None, hidden_states=None):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states


# Build your torch or tf model class here
class FedNLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(config.model_type, config.model_type)
        self.lm_head = BertLMPredictionHead(self.model.encoder.config)

        self.task = config.task
        self.pretrain_task = config.pretrain_task
        self.pt_cfg = self.model.encoder.config
        self.vocab_size = self.pt_cfg.vocab_size
        self.hidden_size = self.pt_cfg.hidden_size
        self.dropout_prob = self.pt_cfg.hidden_dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        self.label_smoothing = config.label_smoothing
        self.padding_idx = config.pad_token_id
        self.classifier = nn.Linear(self.hidden_size, config.num_labels) \
            if config.num_labels is not None else None

        # for eval generation
        self.model.config.decoder_start_token_id = config.bos_token_id
        self.model.config.eos_token_id = config.eos_token_id
        self.model.config.pad_token_id = config.pad_token_id
        self.model.config.vocab_size = self.pt_cfg.vocab_size
        self.model.config.max_length = config.max_length
        self.model.config.min_length = config.min_length
        self.model.config.no_repeat_ngram_size = config.no_repeat_ngram_size
        self.model.config.length_penalty = config.length_penalty
        self.model.config.num_beams = config.num_beams

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        start_positions=None,
        end_positions=None,
        labels=None,
    ):
        enc_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        if self.task == 'pretrain':
            if self.pretrain_task == 'mlm':
                logits = self.lm_head(enc_outputs.last_hidden_state)
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
                loss = masked_lm_loss

            elif self.pretrain_task == 'denoise':
                dec_outputs = self.model.decoder.bert(
                    input_ids=labels,
                    encoder_hidden_states=enc_outputs.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                )
                dec_hidden_states = dec_outputs.last_hidden_state
                logits = self.model.decoder.cls(dec_hidden_states)[:, :-1, :]
                loss_fct = CrossEntropyLoss(ignore_index=self.padding_idx)
                loss = loss_fct(logits.contiguous().view(-1, self.vocab_size),
                                labels[:, 1:].contiguous().view(-1))

        elif self.task in {'imdb', 'agnews'}:
            pooled_output = self.dropout(enc_outputs.pooler_output)
            logits = self.classifier(pooled_output)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        elif self.task in {'squad', 'newsqa'}:
            logits = self.classifier(enc_outputs.last_hidden_state)
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

        elif self.task in {'cnndm', 'msqg'}:
            dec_outputs = self.model.decoder.bert(
                input_ids=labels,
                encoder_hidden_states=enc_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
            dec_hidden_states = dec_outputs.last_hidden_state
            logits = self.model.decoder.cls(dec_hidden_states)[:, :-1, :]

            num_tokens = labels[:, 1:].ne(self.padding_idx).sum().item()
            label_smoothing = self.label_smoothing if self.training else 0.0
            if label_smoothing > 0:
                loss_fct = LabelSmoothingLoss(
                    label_smoothing,
                    self.vocab_size,
                    ignore_index=self.padding_idx,
                ).to(logits.device)
                loss = loss_fct(F.log_softmax(logits.contiguous().view(-1, self.vocab_size), dim=-1),
                                labels[:, 1:].contiguous().view(-1)) / num_tokens
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.padding_idx)
                loss = loss_fct(logits.contiguous().view(-1, self.vocab_size),
                                labels[:, 1:].contiguous().view(-1))

        return ModelOutput(loss=loss, logits=logits)


def call_fednlp_model(model_config, local_data):
    if model_config.type == 'fednlp_model':
        model = FedNLPModel(model_config)
        return model


register_model('fednlp_model', call_fednlp_model)
