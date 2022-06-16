import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from federatedscope.nlp.model.model import ModelOutput


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.segment_embeddings = nn.Embedding(config.segment_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.position_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, segment_ids, position_ids):
        token_embeddings = self.token_embeddings(token_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = self.dropout(self.layernorm(embeddings))
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()

        self.num_attn_heads = config.num_attn_heads
        self.attn_head_size = config.hidden_size // self.num_attn_heads
        self.all_head_size = self.attn_head_size * self.num_attn_heads

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Linear(self.all_head_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attn_mask):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = query.view(hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        key = key.view(hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)
        value = value.view(hidden_states.size(0), -1, self.num_attn_heads, self.attn_head_size).permute(0, 2, 1, 3)

        attn_mask = attn_mask[:, None, None, :]
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attn_head_size)
        attn_score += attn_mask * -10000.0
        attn_prob = self.attn_dropout(self.softmax(attn_score))

        context = torch.matmul(attn_prob, value)
        context = context.permute(0, 2, 1, 3).contiguous().view(value.size(0), -1, self.all_head_size)
        context = self.hidden_dropout(self.dense(context))
        output = self.layernorm(hidden_states + context)
        return output, attn_score


class BertFeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(BertFeedForwardNetwork, self).__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.ffn_hidden_size)
        self.activation = gelu
        self.dense2 = nn.Linear(config.ffn_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        output = self.activation(self.dense1(hidden_states))
        output = self.dropout(self.dense2(output))
        output = self.layernorm(hidden_states + output)
        return output


class BertTransformerBlock(nn.Module):
    def __init__(self, config):
        super(BertTransformerBlock, self).__init__()

        self.attention = BertAttention(config)
        self.ffn = BertFeedForwardNetwork(config)

    def forward(self, hidden_states, attn_mask):
        attn_output, attn_score = self.attention(hidden_states, attn_mask)
        output = self.ffn(attn_output)
        return output, attn_score
        # return output, attn_output


class BertClsPooler(nn.Module):
    def __init__(self, config):
        super(BertClsPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        output = self.activation(self.dense(hidden_states))
        output = self.dropout(output)
        return output


class BertMaskedLMHead(nn.Module):
    def __init__(self, config, embedding_weights):
        super(BertMaskedLMHead, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = gelu
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.lm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.lm_decoder.weight = embedding_weights

    def forward(self, hidden_states):
        hidden_states = self.layernorm(self.activation(self.dense(hidden_states)))
        output = self.lm_decoder(hidden_states) + self.lm_bias
        return output


class BertSingle(nn.Module):
    def __init__(self, config, use_lm=False):
        super(BertSingle, self).__init__()

        self.use_lm = use_lm
        self.embeddings = BertEmbedding(config)
        self.encoder = nn.ModuleList([BertTransformerBlock(config) for _ in range(config.num_layers)])

        if self.use_lm:
            self.lm_head = BertMaskedLMHead(config, self.embeddings.token_embeddings.weight)
        self._init_weights()

    def forward(self, token_ids, segment_ids, position_ids, attn_mask):
        all_attn_outputs, all_ffn_outputs = [], []
        output = self.embeddings(token_ids, segment_ids, position_ids)
        all_ffn_outputs.append(output)

        for layer in self.encoder:
            output, attn_output = layer(output, attn_mask)
            all_attn_outputs.append(attn_output)
            all_ffn_outputs.append(output)

        if self.use_lm:
            output = self.lm_head(output)
        return output, all_attn_outputs, all_ffn_outputs

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


class Bert(nn.Module):
    def __init__(self, config, task, return_hid=False):
        super(Bert, self).__init__()

        self.task = task
        self.return_hid = return_hid
        self.embeddings = BertEmbedding(config)
        self.encoder = nn.ModuleList([BertTransformerBlock(config) for _ in range(config.num_layers)])

        if task == 'squad':
            self.num_classes = 2
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self._init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, **kwargs):
        all_attn_outputs, all_ffn_outputs = [], []
        position_ids = torch.LongTensor([[i for i in range(len(input_ids[0]))]
                                        for _ in range(len(input_ids))]).to(input_ids.device)
        output = self.embeddings(input_ids, token_type_ids, position_ids)
        all_ffn_outputs.append(output)
        for layer in self.encoder:
            output, attn_output = layer(output, attention_mask)
            all_attn_outputs.append(attn_output)
            all_ffn_outputs.append(output)

        if self.task == 'squad':
            logits = self.classifier(output)
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

            return ModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,
                attentions=None,
            )

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
