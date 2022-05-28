import os
import os.path as osp
import logging
import torch
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


def create_cnndm_dataset(root, split, tokenizer, max_src_len, max_tgt_len, model_type, cache_dir=''):
    logger.info('Preprocessing {} {} dataset'.format('cnndm', split))
    cache_file = osp.join(cache_dir, '_'.join(['cnndm', split, str(max_src_len), str(max_tgt_len), model_type]) + '.pt')
    if osp.exists(cache_file):
        logger.info('Loading cache file from \'{}\''.format(cache_file))
        cache_data = torch.load(cache_file)
        src_examples = cache_data['src_examples']
        tgt_examples = cache_data['tgt_examples']
        src_encoded = cache_data['src_encoded']
        tgt_encoded = cache_data['tgt_encoded']
    else:
        src_examples, tgt_examples = create_cnndm_examples(root, split)
        tgt_examples = preprocess_tgt_examples(tgt_examples)
        src_encoded = tokenizer(src_examples, padding=True, truncation=True, max_length=max_src_len, return_tensors='pt')
        tgt_encoded = tokenizer(tgt_examples, padding=True, truncation=True, max_length=max_tgt_len, return_tensors='pt',
                                add_special_tokens=False)

        if cache_dir:
            logger.info('Saving cache file to \'{}\''.format(cache_file))
            os.makedirs(cache_dir, exist_ok=True)
            torch.save({'src_examples': src_examples,
                        'tgt_examples': tgt_examples,
                        'src_encoded': src_encoded,
                        'tgt_encoded': tgt_encoded}, cache_file)

    token_ids = src_encoded.input_ids
    token_type_ids = src_encoded.token_type_ids
    attention_mask = src_encoded.attention_mask
    labels = tgt_encoded.input_ids

    dataset = TensorDataset(token_ids, token_type_ids, attention_mask, labels)
    return dataset, (src_encoded, tgt_encoded), (src_examples, tgt_examples)
