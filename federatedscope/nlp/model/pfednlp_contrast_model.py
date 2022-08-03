import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertForPreTraining
from federatedscope.register import register_model
from federatedscope.nlp.loss import LabelSmoothingLoss
from federatedscope.nlp.model.decoder import TransformerDecoder


class ModelOutput(object):
    def __init__(self, loss=None, logits=None, hidden_states=None, example_indices=None):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.example_indices = example_indices


class ContrastiveHead(nn.Module):
    def __init__(self, input_dim, inner_dim, out_dim, dropout_prob):
        super().__init__()

        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_prj = nn.Linear(inner_dim, out_dim)

    def forward(self, x):
        x = self.dense(self.dropout(x))
        x = torch.tanh(x)
        x = self.out_prj(self.dropout(x))
        return x


# Build your torch or tf model class here
class PFedNLPContrastModel(nn.Module):
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
        self.contrast_head = ContrastiveHead(input_dim=self.hidden_size,
                                             inner_dim=self.hidden_size,
                                             out_dim=self.hidden_size,
                                             dropout_prob=self.dropout_prob)
        self._init_params()

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
        pretrain_task=None,
        in_contrast_prepare=None,
        contrast_monitor=None,
        example_indices=None,
        client_id=None,
        config=None,  # client specific config
    ):
        # print(example_indices)
        if in_contrast_prepare:
            assert pretrain_task == 'denoise'
            if contrast_monitor.stat == 0:  # return enc_hidden_states
                self.eval()
                with torch.no_grad():
                    example_indices = [k for k in example_indices if k.item() < config.federate.num_contrast]
                    if len(example_indices) == 0:
                        return ModelOutput()

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
                    return ModelOutput(hidden_states=outputs.last_hidden_state)

            elif contrast_monitor.stat == 1:  # return dec_hidden_states & dec_out
                self.eval()
                with torch.no_grad():
                    example_indices = [k for k in example_indices if k.item() in contrast_monitor.synth_tokens]
                    if len(example_indices) == 0:
                        return ModelOutput()

                    example_indices = torch.stack(example_indices)
                    synth_input_ids = torch.stack([contrast_monitor.synth_tokens[k.item()] for k in example_indices]).to(config.device)
                    enc_hidden = torch.stack([contrast_monitor.enc_hidden[k.item()] for k in example_indices]).to(config.device)

                    dec_state = self.decoder.init_decoder_state(synth_input_ids, enc_hidden)
                    dec_outputs, _ = self.decoder(synth_input_ids[:, :-1], enc_hidden, dec_state)
                    logits = self.generator[0](dec_outputs.view(-1, dec_outputs.size(-1)))
                    preds = logits.view(-1, dec_outputs.size(1), self.vocab_size).argmax(dim=-1)
                    dec_hidden = self.contrast_head(dec_outputs).mean(dim=1)
                    return ModelOutput(logits=preds, hidden_states=dec_hidden, example_indices=example_indices)

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
            if pretrain_task == 'mlm':
                logits = self.encoder.cls.predictions(outputs.last_hidden_state)
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
                loss = masked_lm_loss

            elif pretrain_task == 'denoise':
                # denoising loss
                target_ids = labels
                dec_state = self.decoder.init_decoder_state(input_ids, outputs.last_hidden_state)
                dec_outputs, _ = self.decoder(target_ids[:, :-1], outputs.last_hidden_state, dec_state)
                logits = self.generator(dec_outputs.view(-1, dec_outputs.size(-1)))
                loss_fct = nn.NLLLoss(ignore_index=self.padding_idx, reduction='sum')
                num_tokens = target_ids[:, 1:].ne(self.padding_idx).sum().item()
                denoise_loss = loss_fct(logits, target_ids[:, 1:].contiguous().view(-1)) / num_tokens
                loss = denoise_loss

                # decoder kd loss with dec_out
                contrast_dec_outputs = None
                if contrast_monitor is not None and config.federate.train_dec_out:
                    dec_loss_fct = nn.NLLLoss()
                    dec_out_loss, num_items = 0, 0
                    example_indices = [k for k in example_indices if k.item() in contrast_monitor.synth_tokens]

                    if len(example_indices) > 0:
                        example_indices = torch.stack(example_indices)
                        synth_input_ids = torch.stack([contrast_monitor.synth_tokens[k.item()] for k in example_indices]).to(config.device)
                        contrast_enc_hidden = torch.stack([contrast_monitor.enc_hidden[k.item()] for k in example_indices]).to(config.device)
                        contrast_dec_state = self.decoder.init_decoder_state(synth_input_ids, contrast_enc_hidden)
                        contrast_dec_outputs, _ = self.decoder(synth_input_ids[:, :-1], contrast_enc_hidden, contrast_dec_state)
                        pred_logits = self.generator(contrast_dec_outputs.view(-1, contrast_dec_outputs.size(-1)))

                        group_ids, dec_out = contrast_monitor.group_ids, contrast_monitor.dec_out
                        inside_weight, outside_weight = config.aggregator.inside_weight, config.aggregator.outside_weight
                        for cid in group_ids:
                            if dec_out[cid] is None or cid == client_id:
                                continue
                            cur_compare_preds = torch.stack([dec_out[cid][k.item()] for k in example_indices]).to(config.device)
                            cur_loss = dec_loss_fct(pred_logits, cur_compare_preds.contiguous().view(-1))

                            if group_ids[cid] == group_ids[client_id]:
                                dec_out_loss += inside_weight * cur_loss
                                num_items += inside_weight
                            else:
                                dec_out_loss += outside_weight * cur_loss
                                num_items += outside_weight

                        if num_items > 0:
                            # print('Denoise loss: {:.4f}\tContrast loss: {:.4f}'
                            #       .format(denoise_loss.detach().cpu(), contrast_loss.detach().cpu() / num_contrast_item))
                            loss += dec_out_loss / num_items

                # decoder contrastive loss with dec_hidden
                if contrast_monitor is not None and config.federate.train_dec_hidden:
                    example_indices = [k for k in example_indices if k.item() in contrast_monitor.synth_tokens]
                    if len(example_indices) > 0:
                        example_indices = torch.stack(example_indices)
                        if contrast_dec_outputs is None:
                            synth_input_ids = torch.stack([contrast_monitor.synth_tokens[k.item()] for k in example_indices]).to(config.device)
                            contrast_enc_hidden = torch.stack([contrast_monitor.enc_hidden[k.item()] for k in example_indices]).to(config.device)
                            contrast_dec_state = self.decoder.init_decoder_state(synth_input_ids, contrast_enc_hidden)
                            contrast_dec_outputs, _ = self.decoder(synth_input_ids[:, :-1], contrast_enc_hidden, contrast_dec_state)
                        cur_dec_hidden = self.contrast_head(contrast_dec_outputs)

                        group_ids, all_dec_hiddens = contrast_monitor.group_ids, contrast_monitor.dec_hidden
                        group2client_ids = {}
                        for k, v in group_ids.items():
                            if group2client_ids.get(v, None) is None:
                                group2client_ids[v] = []
                            group2client_ids[v].append(k)
                        group_dec_hiddens = {}
                        for k, v in group2client_ids.items():
                            group_dec_hiddens[k] = torch.stack([torch.stack([
                                all_dec_hiddens[cid][k.item()] for k in example_indices]) for cid in v]).mean(dim=0)

                        cur_group_id = group_ids[client_id]
                        sim_hiddens = group_dec_hiddens[cur_group_id]
                        sim_matrix = F.cosine_similarity(cur_dec_hidden, sim_hiddens, dim=-1)
                        dissim_hiddens = [v for k, v in group_dec_hiddens.items() if k != cur_group_id]
                        dissim_matrix = None
                        if len(dissim_hiddens) > 0:
                            dissim_hiddens = torch.stack(dissim_hiddens, dim=1)
                            dissim_matrix = F.cosine_similarity(cur_dec_hidden.unsqueeze(1), dissim_hiddens)

                        temperature = config.federate.contrast_temp
                        if dissim_matrix is not None:
                            nominator = torch.exp(sim_matrix / temperature)
                            denominator = torch.exp(dissim_matrix / temperature)
                            dec_hidden_loss = -torch.log(nominator / torch.sum(denominator, dim=-1))
                        else:
                            dec_hidden_loss = sim_matrix / temperature
                        dec_hidden_loss = torch.sum(dec_hidden_loss) / sim_matrix.size(0)
                        loss += dec_hidden_loss

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
        )

    def _init_params(self):
        self.encoder._init_weights(self.contrast_head.dense)
        self.encoder._init_weights(self.contrast_head.out_prj)
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


def call_pfednlp_contrast_model(model_config, local_data):
    if model_config.type == 'pfednlp_contrast_model':
        model = PFedNLPContrastModel(model_config)
        return model


register_model('pfednlp_contrast_model', call_pfednlp_contrast_model)
