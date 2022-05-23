import os
import os.path as osp
import random
import logging
import torch
from torch.utils.data.dataset import TensorDataset

logger = logging.getLogger(__name__)


def read_file(path):
    with open(path) as f:
        data = f.readline()
    return data


def create_imdb_examples(root, split):
    examples = []
    pos_files = os.listdir(osp.join(root, split, 'pos'))
    for file in pos_files:
        path = osp.join(root, split, 'pos', file)
        data = read_file(path)
        examples.append((data, 1))
    neg_files = os.listdir(osp.join(root, split, 'neg'))
    for file in neg_files:
        path = osp.join(root, split, 'neg', file)
        data = read_file(path)
        examples.append((data, 0))
    random.shuffle(examples)

    if split == 'train':
        num_train_samples = int(0.9 * len(examples))
        return examples[:num_train_samples], examples[num_train_samples:]
    elif split == 'test':
        return examples


def create_imdb_dataset(root, split, tokenizer, max_seq_len):
    logger.info('Preprocessing {} {} dataset'.format('imdb', split))
    examples = create_imdb_examples(root, split)

    def _create_dataset(examples_):
        texts = [ex[0] for ex in examples_]
        labels = [ex[1] for ex in examples_]
        encoded_inputs = tokenizer(texts, add_special_tokens=True, padding=True, truncation=True, max_length=max_seq_len)

        token_ids = torch.LongTensor(encoded_inputs.input_ids)
        token_type_ids = torch.LongTensor(encoded_inputs.token_type_ids)
        attention_mask = torch.LongTensor(encoded_inputs.attention_mask)
        labels = torch.LongTensor(labels)

        dataset = TensorDataset(token_ids, token_type_ids, attention_mask, labels)
        return dataset, encoded_inputs, examples_

    if split == 'train':
        return _create_dataset(examples[0]), _create_dataset(examples[1])
    elif split == 'test':
        return _create_dataset(examples)
