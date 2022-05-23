import os
import csv
import logging
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

glue_labels = {
    'mrpc': ['0', '1'],
    'mnli': ['contradiction', 'entailment', 'neutral'],
    'mnli-mm': ['contradiction', 'entailment', 'neutral'],
    'ax': ['contradiction', 'entailment', 'neutral'],
    'cola': ['0', '1'],
    'sst-2': ['0', '1'],
    'sts-b': [None],
    'qqp': ['0', '1'],
    'qnli': ['entailment', 'not_entailment'],
    'rte': ['entailment', 'not_entailment'],
    'wnli': ['0', '1']
}


class GlueExample(object):
    def __init__(self, text_a, text_b, label, id):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.id = id


def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        data = []
        for line in reader:
            data.append(line)
        return data


def create_glue_examples(glue_dir, task, split):
    assert split in ['train', 'dev', 'test']
    file_name = split + '.tsv'
    if task == 'mrpc':
        data_path = os.path.join(glue_dir, 'MRPC', file_name)
        text_a_id, text_b_id, label_id = [3, 4, 0] if split != 'test' else [3, 4, None]
    elif task == 'mnli':
        data_path = os.path.join(glue_dir, 'MNLI', file_name if split == 'train' else split + '_matched.tsv')
        text_a_id, text_b_id, label_id = [8, 9, -1] if split != 'test' else [8, 9, None]
    elif task == 'mnli-mm':
        data_path = os.path.join(glue_dir, 'MNLI', file_name if split == 'train' else split + '_mismatched.tsv')
        text_a_id, text_b_id, label_id = [8, 9, -1] if split != 'test' else [8, 9, None]
    elif task == 'ax':
        data_path = os.path.join(glue_dir, 'MNLI', 'diagnostic.tsv')
        text_a_id, text_b_id, label_id = [1, 2, None]
    elif task == 'cola':
        data_path = os.path.join(glue_dir, 'CoLA', file_name)
        text_a_id, text_b_id, label_id = [3, None, 1] if split != 'test' else [1, None, None]
    elif task == 'sst-2':
        data_path = os.path.join(glue_dir, 'SST-2', file_name)
        text_a_id, text_b_id, label_id = [0, None, 1] if split != 'test' else [1, None, None]
    elif task == 'sts-b':
        data_path = os.path.join(glue_dir, 'STS-B', file_name)
        text_a_id, text_b_id, label_id = [7, 8, -1] if split != 'test' else [7, 8, None]
    elif task == 'qqp':
        data_path = os.path.join(glue_dir, 'QQP', file_name)
        text_a_id, text_b_id, label_id = [3, 4, 5] if split != 'test' else [1, 2, None]
    elif task == 'qnli':
        data_path = os.path.join(glue_dir, 'QNLI', file_name)
        text_a_id, text_b_id, label_id = [1, 2, -1] if split != 'test' else [1, 2, None]
    elif task == 'rte':
        data_path = os.path.join(glue_dir, 'RTE', file_name)
        text_a_id, text_b_id, label_id = [1, 2, -1] if split != 'test' else [1, 2, None]
    elif task == 'wnli':
        data_path = os.path.join(glue_dir, 'WNLI', file_name)
        text_a_id, text_b_id, label_id = [1, 2, -1] if split != 'test' else [1, 2, None]
    else:
        raise KeyError('task \'{}\' is not valid'.format(task))

    labels = glue_labels[task]
    label_map = {label: i for i, label in enumerate(labels)}
    data = read_tsv(data_path)

    examples = []
    for i, line in enumerate(data):
        if i == 0 and (split == 'test' or (split != 'test' and task != 'cola')):
            continue
        text_a = line[text_a_id]
        text_b = line[text_b_id] if text_b_id is not None else None
        if split == 'test':
            label = None
        else:
            label = line[label_id]
            label = float(label) if task == 'sts-b' else label_map[label]

        id = int(line[0]) if split == 'test' else None
        examples.append(GlueExample(text_a, text_b, label, id))
    return examples


def create_glue_dataset(glue_dir, task, tokenizer, max_seq_len, split, cache_dir=''):
    cache_file = os.path.join(cache_dir, 'glue', '_'.join([task, split, str(max_seq_len)]))
    if tokenizer.do_lower_case:
        cache_file = os.path.join(cache_dir, 'glue', '_'.join([task, split, str(max_seq_len), 'lowercase']))

    logger.info('Preprocessing {} {} datasets'.format(task, split))
    if os.path.exists(cache_file):
        logger.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        examples = create_glue_examples(glue_dir, task, split)
        texts_a = [example.text_a for example in examples]
        texts_b = [example.text_b for example in examples]
        texts_b = None if texts_b[0] is None else texts_b
        encoded_inputs = tokenizer(texts_a, texts_b, add_special_tokens=True, padding=True, truncation=True, max_length=max_seq_len)

        if cache_dir:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.join(cache_dir, 'glue')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples, 'encoded_inputs': encoded_inputs}, cache_file)

    if split == 'train':
        # num_train_samples0 = int(0.02 * len(examples))
        num_train_samples = int(0.8 * len(examples))
        train_token_ids = torch.tensor(encoded_inputs.input_ids[:num_train_samples], dtype=torch.long)
        train_token_type_ids = torch.tensor(encoded_inputs.token_type_ids[:num_train_samples], dtype=torch.long)
        train_attn_mask = torch.tensor(encoded_inputs.attention_mask[:num_train_samples], dtype=torch.long)
        train_labels = torch.tensor([example.label for example in examples[:num_train_samples]], dtype=torch.float
                                     if task == 'sts-b' else torch.long)
        train_dataset = TensorDataset(train_token_ids, train_token_type_ids, train_attn_mask, train_labels)

        dev_token_ids = torch.tensor(encoded_inputs.input_ids[num_train_samples:], dtype=torch.long)
        dev_token_type_ids = torch.tensor(encoded_inputs.token_type_ids[num_train_samples:], dtype=torch.long)
        dev_attn_mask = torch.tensor(encoded_inputs.attention_mask[num_train_samples:], dtype=torch.long)
        dev_labels = torch.tensor([example.label for example in examples[num_train_samples:]], dtype=torch.float
                                   if task == 'sts-b' else torch.long)
        dev_dataset = TensorDataset(dev_token_ids, dev_token_type_ids, dev_attn_mask, dev_labels)

        return train_dataset, dev_dataset

    else:
        token_ids = torch.tensor(encoded_inputs.input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(encoded_inputs.token_type_ids, dtype=torch.long)
        attn_mask = torch.tensor(encoded_inputs.attention_mask, dtype=torch.long)

        if split == 'test':
            ids = torch.tensor([example.id for example in examples], dtype=torch.long)
            dataset = TensorDataset(token_ids, token_type_ids, attn_mask, ids)
        else:  # split == 'dev'
            labels = torch.tensor([example.label for example in examples], dtype=torch.float if task == 'sts-b' else torch.long)
            dataset = TensorDataset(token_ids, token_type_ids, attn_mask, labels)

        return dataset


def load_glue_data(config):
    tokenizer = config.data.tokenizer
    max_seq_len = 128

    qqp_train, qqp_dev = create_glue_dataset(glue_dir, 'qqp', tokenizer, max_seq_len, 'train')
    qqp_test = create_glue_dataset(glue_dir, 'qqp', tokenizer, max_seq_len, 'dev')
    # qnli_train, qnli_dev = create_glue_dataset(glue_dir, 'qnli', tokenizer, max_seq_len, 'train')
    # qnli_test = create_glue_dataset(glue_dir, 'qnli', tokenizer, max_seq_len, 'dev')
    # sst2_train, sst2_dev = create_glue_dataset(glue_dir, 'sst-2', tokenizer, max_seq_len, 'train')
    # sst2_test = create_glue_dataset(glue_dir, 'sst-2', tokenizer, max_seq_len, 'dev')
    # mrpc_train, mrpc_dev = create_glue_dataset(glue_dir, 'mrpc', tokenizer, max_seq_len, 'train')
    # mrpc_test = create_glue_dataset(glue_dir, 'mrpc', tokenizer, max_seq_len, 'dev')

    train_data = [qqp_train]
    dev_data = [qqp_dev]
    test_data = [qqp_test]
    # train_data = [qnli_train]
    # dev_data = [qnli_dev]
    # test_data = [qnli_test]
    # train_data = [sst2_train]
    # dev_data = [sst2_dev]
    # test_data = [sst2_test]
    # train_data = [mrpc_train]
    # dev_data = [mrpc_dev]
    # test_data = [mrpc_test]

    data_dict = dict()
    for client_idx in range(1, config.federate.client_num + 1):
        dataloader_dict = {
            'train': DataLoader(train_data[client_idx - 1], config.data.batch_size, shuffle=config.data.shuffle),
            'val': DataLoader(dev_data[client_idx - 1], config.data.batch_size, shuffle=False),
            'test': DataLoader(test_data[client_idx - 1], config.data.batch_size, shuffle=False)
        }
        data_dict[client_idx] = dataloader_dict

    return data_dict, config


def call_my_data(config):
    if config.data.type == "mydata":
        data, modified_config = load_glue_data(config)
        return data, modified_config
