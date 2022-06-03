import os
import os.path as osp
import logging
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from torch.utils.data.dataset import TensorDataset

logger = logging.getLogger(__name__)


def read_file(path):
    with open(path) as f:
        data = [line.strip() for line in f]
    return data


def create_cnndm_examples(root, split):
    src_file = osp.join(root, split + '.source')
    tgt_file = osp.join(root, split + '.target')
    src_examples = read_file(src_file)
    tgt_examples = read_file(tgt_file)
    return src_examples, tgt_examples


def preprocess_tgt_examples(examples, bos='[unused0]', eos='[unused1]', eoq='[unused2]'):
    new_examples = []
    for e in examples:
        sents = [s[:-1] for s in sent_tokenize(e)]
        e = '{} '.format(bos) + ' {} '.format(eoq).join(sents) + ' {}'.format(eos)
        new_examples.append(e)
    return new_examples


def create_cnndm_dataset(root, split, tokenizer, max_src_len, max_tgt_len, model_type, raw_cache_dir=''):
    logger.info('Preprocessing {} {} dataset'.format('cnndm', split))
    cache_dir = osp.join(raw_cache_dir, 'cnndm', '_'.join([str(max_src_len), str(max_tgt_len), model_type]), split)

    src_examples, tgt_examples = create_cnndm_examples(root, split)
    if osp.exists(cache_dir):
        logger.info('Loading cache file from \'{}\''.format(cache_dir))
        token_ids = np.memmap(filename=osp.join(cache_dir, 'token_ids.memmap'),
                              shape=(len(src_examples), max_src_len),
                              mode='r',
                              dtype=np.int64)
        token_type_ids = np.memmap(filename=osp.join(cache_dir, 'token_type_ids.memmap'),
                                   shape=(len(src_examples), max_src_len),
                                   mode='r',
                                   dtype=np.int64)
        attention_mask = np.memmap(filename=osp.join(cache_dir, 'attention_mask.memmap'),
                                   shape=(len(src_examples), max_src_len),
                                   mode='r',
                                   dtype=np.int64)
        labels = np.memmap(filename=osp.join(cache_dir, 'labels.memmap'),
                           shape=(len(src_examples), max_tgt_len),
                           mode='r',
                           dtype=np.int64)
    else:
        tgt_examples = preprocess_tgt_examples(tgt_examples)
        src_encoded = tokenizer(src_examples, padding='max_length', truncation=True, max_length=max_src_len,
                                return_tensors='pt')
        tgt_encoded = tokenizer(tgt_examples, padding='max_length', truncation=True, max_length=max_tgt_len,
                                return_tensors='pt', add_special_tokens=False)

        if raw_cache_dir:
            logger.info('Saving cache file to \'{}\''.format(cache_dir))
            os.makedirs(cache_dir, exist_ok=True)
            token_ids = np.memmap(filename=osp.join(cache_dir, 'token_ids.memmap'),
                                  shape=(len(src_examples), max_src_len),
                                  mode='w+',
                                  dtype=np.int64)
            token_type_ids = np.memmap(filename=osp.join(cache_dir, 'token_type_ids.memmap'),
                                       shape=(len(src_examples), max_src_len),
                                       mode='w+',
                                       dtype=np.int64)
            attention_mask = np.memmap(filename=osp.join(cache_dir, 'attention_mask.memmap'),
                                       shape=(len(src_examples), max_src_len),
                                       mode='w+',
                                       dtype=np.int64)
            labels = np.memmap(filename=osp.join(cache_dir, 'labels.memmap'),
                               shape=(len(src_examples), max_tgt_len),
                               mode='w+',
                               dtype=np.int64)

            for i in range(len(src_examples)):
                token_ids[i] = src_encoded.input_ids[i]
                token_type_ids[i] = src_encoded.token_type_ids[i]
                attention_mask[i] = src_encoded.attention_mask[i]
                labels[i] = tgt_encoded.input_ids[i]

    token_ids = torch.from_numpy(token_ids)
    token_type_ids = torch.from_numpy(token_type_ids)
    attention_mask = torch.from_numpy(attention_mask)
    labels = torch.from_numpy(labels)

    dataset = TensorDataset(token_ids, token_type_ids, attention_mask, labels)
    return dataset, None, None
