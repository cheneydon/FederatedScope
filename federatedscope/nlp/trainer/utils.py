import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert import BertTokenizerFast


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ContrastiveMonitor(object):
    def __init__(self, stat=0, enc_hidden=None, mlm_head_params=None, synth_tokens=None,
                 dec_hidden=None, dec_out=None, group_ids=None):
        self.stat = stat
        self.enc_hidden = enc_hidden
        self.mlm_head_params = mlm_head_params
        self.synth_tokens = synth_tokens
        self.dec_hidden = dec_hidden
        self.dec_out = dec_out
        self.group_ids = group_ids

    def update_stat(self, status):
        self.stat = status

    def update_mlm_head_params(self, mlm_head_params):
        self.mlm_head_params = mlm_head_params

    def update_group_ids(self, group_ids):
        self.group_ids = group_ids

    def update_enc_hidden(self, enc_hidden, k=None):
        if k is None:
            self.enc_hidden = enc_hidden
        else:
            if self.enc_hidden is None:
                self.enc_hidden = {}
            self.enc_hidden[k] = enc_hidden

    def update_synth_tokens(self, synth_tokens, k=None):
        if k is None:
            self.synth_tokens = synth_tokens
        else:
            if self.synth_tokens is None:
                self.synth_tokens = {}
            self.synth_tokens[k] = synth_tokens

    def update_dec_hidden(self, dec_hidden, k=None):
        if k is None:
            self.dec_hidden = dec_hidden
        else:
            if self.dec_hidden is None:
                self.dec_hidden = {}
            self.dec_hidden[k] = dec_hidden

    def update_dec_out(self, dec_out, k=None):
        if k is None:
            self.dec_out = dec_out
        else:
            if self.dec_out is None:
                self.dec_out = {}
            self.dec_out[k] = dec_out

    def reset(self):
        self.stat = 0
        self.enc_hidden = None
        self.mlm_head_params = None
        self.synth_tokens = None
        self.dec_hidden = None
        self.dec_out = None
        self.group_ids = None


def setup_tokenizer(model_type):
    bos_token, eos_token, eoq_token = '[unused0]', '[unused1]', '[unused2]'
    try:
        tokenizer = BertTokenizerFast.from_pretrained(
            model_type,
            additional_special_tokens=[bos_token, eos_token, eoq_token],
            skip_special_tokens=True,
            local_files_only=True,
        )
    except:
        tokenizer = BertTokenizerFast.from_pretrained(
            model_type,
            additional_special_tokens=[bos_token, eos_token, eoq_token],
            skip_special_tokens=True,
        )
    tokenizer.bos_token = bos_token
    tokenizer.eos_token = eos_token
    tokenizer.eoq_token = eoq_token
    tokenizer.bos_token_id = tokenizer.vocab[bos_token]
    tokenizer.eos_token_id = tokenizer.vocab[eos_token]
    tokenizer.eoq_token_id = tokenizer.vocab[eoq_token]
    tokenizer.symbols = {'BOS': tokenizer.bos_token_id, 'EOS': tokenizer.eos_token_id,
                         'PAD': tokenizer.pad_token_id, 'EOQ': tokenizer.eoq_token_id}
    return tokenizer
