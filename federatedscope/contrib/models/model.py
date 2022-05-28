import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert import BertModel
from federatedscope.contrib.models.decoder import TransformerDecoder


class ModelOutput(object):
    def __init__(self, loss, logits, hidden_states, attentions):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions


# Build your torch or tf model class here
class MyModel(nn.Module):
    def __init__(self, config):  # server config
        super().__init__()

        self.bert = BertModel.from_pretrained(config.bert_type)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout_prob = self.bert.config.hidden_dropout_prob

        # For NLU
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.ModuleDict({'classifier_{}'.format(k): nn.Linear(self.hidden_size, k)
                                         for k in config.num_labels})

        # For NLG
        self.vocab_size = self.bert.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        tgt_embeddings.weight = copy.deepcopy(self.bert.embeddings.word_embeddings.weight)
        self.decoder = TransformerDecoder(
            num_layers=config.num_dec_layers,
            d_model=self.hidden_size,
            heads=self.bert.config.num_attention_heads,
            d_ff=config.dec_d_ffn,
            dropout=config.dec_dropout_prob,
            embeddings=tgt_embeddings,
        )
        self.decoder.embeddings = tgt_embeddings

        self.generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.vocab_size),
            nn.LogSoftmax(dim=-1),
        )
        self.generator[0].weight = self.decoder.embeddings.weight

        self._init_decoder_params()

    def forward(
        self,
        input_ids=None,
        target_ids=None,
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
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        task = config.data.type
        num_labels = config.data.num_labels

        if task == 'imdb':
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier['classifier_{}'.format(num_labels)](pooled_output)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        elif task == 'squad':
            sequence_output = outputs.last_hidden_state
            logits = self.classifier['classifier_{}'.format(num_labels)](sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
            logits = (start_logits, end_logits)

            loss = None
            if start_positions is not None and end_positions is not None:
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

        elif task == 'cnndm':
            dec_state = self.decoder.init_decoder_state(input_ids, outputs.last_hidden_state)
            decoder_outputs, state = self.decoder(target_ids[:, :-1], outputs.last_hidden_state, dec_state)
            loss = None
            logits = decoder_outputs

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


# Instantiate your model class with config and data
def ModelBuilder(model_config, local_data):
    model_type = 1

    if model_type == 1:
        model = MyModel(model_config)
        return model

    if model_type == 2:
        from federatedscope.contrib.models.bert import Bert

        class BertBaseConfig(object):
            def __init__(self, lowercase=False):
                self.vocab_size = 28996 if not lowercase else 30522
                self.position_size = 512
                self.segment_size = 2
                self.hidden_size = 768
                self.hidden_dropout_prob = 0.1
                self.num_attn_heads = 12
                self.attn_dropout_prob = 0.1
                self.ffn_hidden_size = 3072
                self.num_layers = 12
                self.pad_token_id = 0
                self.sep_token_id = 102

        config = BertBaseConfig(True)
        model = Bert(config, 'squad')

        import re
        print('Loading pretrained ckpt')
        ckpt_path = '/mnt/dongchenhe.dch/efficient-bert/pretrained_ckpt/bert-base-uncased-pytorch_model.bin'
        raw_state_dict = torch.load(ckpt_path, map_location='cpu')
        new_state_dict = {}
        for n, p in raw_state_dict.items():
            if re.search(r'pooler|cls', n) is not None: continue
            n = re.sub(r'(bert|layer|self)\.', '', n)
            n = re.sub(r'word_embeddings', 'token_embeddings', n)
            n = re.sub(r'token_type_embeddings', 'segment_embeddings', n)
            n = re.sub(r'LayerNorm', 'layernorm', n)
            n = re.sub(r'gamma', 'weight', n)
            n = re.sub(r'beta', 'bias', n)
            n = re.sub(r'attention\.output', 'attention', n)
            n = re.sub(r'intermediate\.dense', 'ffn.dense1', n)
            n = re.sub(r'output\.dense', 'ffn.dense2', n)
            n = re.sub(r'output', 'ffn', n)
            new_state_dict[n] = p
        model_state_dict = model.state_dict()
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)

        return model

    if model_type == 3:
        from federatedscope.contrib.models.bert import Bert

        class BertBaseConfig(object):
            def __init__(self, lowercase=False):
                self.vocab_size = 28996 if not lowercase else 30522
                self.position_size = 512
                self.segment_size = 2
                self.hidden_size = 768
                self.hidden_dropout_prob = 0.1
                self.num_attn_heads = 12
                self.attn_dropout_prob = 0.1
                self.ffn_hidden_size = 3072
                self.num_layers = 12
                self.pad_token_id = 0
                self.sep_token_id = 102

        config = BertBaseConfig(True)
        model = Bert(config, 'squad')

        import re
        path = '/mnt/dongchenhe.dch/efficient-bert/exp/train/bert_base/20220526-194012/ckpt_ep3.bin'
        ckpt = torch.load(path, map_location='cpu')['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for n, p in ckpt.items():
            n = re.sub(r'module\.', '', n)
            new_state_dict[n] = p
        model.load_state_dict(new_state_dict)

        return model


    if model_type == 4:
        class BertBaseConfig(object):
            def __init__(self):
                self.architectures = "BertForMaskedLM"
                self.attention_probs_dropout_prob = 0.1
                self.attn_dropout_prob = self.attention_probs_dropout_prob
                self.gradient_checkpointing = False
                self.hidden_act = "gelu"
                self.hidden_dropout_prob = 0.1
                self.hidden_size = 768
                self.initializer_range = 0.02
                self.intermediate_size = 3072
                self.layer_norm_eps = 1e-12
                self.max_position_embeddings = 512
                self.model_type = "bert"
                self.num_attention_heads = 12
                self.num_hidden_layers = 12
                self.num_attn_heads = self.num_attention_heads
                self.pad_token_id = 0
                self.position_embedding_type = "absolute"
                self.transformers_version = "4.6.0.dev0"
                self.type_vocab_size = 2
                self.use_cache = True
                self.vocab_size = 30522
                self.chunk_size_feed_forward = 0
                self.is_decoder = False
                self.add_cross_attention = False

        config = BertBaseConfig()

        path = '/mnt/dongchenhe.dch/efficient-bert/exp/train/bert_base/20220526-194012/ckpt_ep3.bin'
        ckpt = torch.load(path, map_location='cpu')['state_dict']

        import re
        from collections import OrderedDict
        model = MyModel2(config)

        new_state_dict = {}
        for n, p in ckpt.items():
            n = re.sub(r'module\.', '', n)
            n = re.sub(r'token_embeddings', 'word_embeddings', n)
            n = re.sub(r'segment_embeddings', 'token_type_embeddings', n)
            n = re.sub(r'layernorm', 'LayerNorm', n)
            # n = re.sub(r'encoder', 'encoder.layer', n)
            # n = re.sub(r'attention\.query', 'attention.self.query', n)
            # n = re.sub(r'attention\.key', 'attention.self.key', n)
            # n = re.sub(r'attention\.value', 'attention.self.value', n)
            # n = re.sub(r'attention\.dense', 'attention.output.dense', n)
            # n = re.sub(r'attention\.LayerNorm', 'attention.output.LayerNorm', n)
            n = re.sub(r'ffn\.dense1', 'intermediate.dense', n)
            n = re.sub(r'ffn\.dense2', 'output.dense', n)
            n = re.sub(r'ffn', 'output', n)
            n = re.sub(r'classifier', 'classifier.classifier_2', n)
            new_state_dict[n] = p
        model_state_dict = model.state_dict()
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # model.load_state_dict(OrderedDict(new_state_dict))

        return model


def call_my_net(model_config, local_data):
    if model_config.type == "mynet":
        model = ModelBuilder(model_config, local_data)
        return model
