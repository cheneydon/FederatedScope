from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizerFast
from federatedscope.contrib.data.imdb import create_imdb_dataset
from federatedscope.contrib.data.squad import create_squad_dataset


def load_my_data(config):
    tokenizer = BertTokenizerFast.from_pretrained(config.model.bert_type)

    # imdb
    root = config.data.dir.imdb
    max_seq_len = config.data.max_seq_len.imdb
    imdb_train, imdb_dev = create_imdb_dataset(root, 'train', tokenizer, max_seq_len)
    imdb_test = create_imdb_dataset(root, 'test', tokenizer, max_seq_len)

    # squad
    root = config.data.dir.squad
    max_seq_len = config.data.max_seq_len.squad
    max_query_len = config.data.max_query_len.squad
    trunc_stride = config.data.trunc_stride.squad
    squad_train, squad_dev = create_squad_dataset(root, 'train', tokenizer, max_seq_len, max_query_len, trunc_stride)
    squad_test = create_squad_dataset(root, 'dev', tokenizer, max_seq_len, max_query_len, trunc_stride)

    train_data = [imdb_train, squad_train]  # train_data[i]: (daataset, encoded_inputs, examples)
    dev_data = [imdb_dev, squad_dev]
    test_data = [imdb_test, squad_test]

    data_dict = dict()
    for client_idx in range(1, config.federate.client_num + 1):
        dataloader_dict = {
            'train': {'dataloader': DataLoader(train_data[client_idx - 1][0], config.data.batch_size, shuffle=config.data.shuffle),
                      'encoded': train_data[client_idx - 1][1],
                      'examples': train_data[client_idx - 1][2]},
            'val': {'dataloader': DataLoader(dev_data[client_idx - 1][0], config.data.batch_size, shuffle=False),
                    'encoded': dev_data[client_idx - 1][1],
                    'examples': dev_data[client_idx - 1][2]},
            'test': {'dataloader': DataLoader(test_data[client_idx - 1][0], config.data.batch_size, shuffle=False),
                     'encoded': test_data[client_idx - 1][1],
                     'examples': test_data[client_idx - 1][2]},
        }
        data_dict[client_idx] = dataloader_dict

    return data_dict, config


def call_my_data(config):
    if config.data.type == "mydata":
        data, modified_config = load_my_data(config)
        return data, modified_config
