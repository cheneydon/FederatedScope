import os.path as osp
from torch.utils.data import DataLoader
from federatedscope.contrib.data.imdb import create_imdb_dataset
from federatedscope.contrib.data.squad import create_squad_dataset
from federatedscope.contrib.data.cnndm import create_cnndm_dataset


def load_my_data(config, tokenizer, **kwargs):
    model_type = config.model.bert_type

    # imdb
    # root = config.data.dir.imdb
    # cache_dir = osp.join(config.data.cache_dir, 'imdb') if config.data.cache_dir else ''
    # max_seq_len = config.data.max_seq_len.imdb
    # imdb_batch_size = config.data.all_batch_size.imdb
    # imdb_train, imdb_dev = create_imdb_dataset(root, 'train', tokenizer, max_seq_len,
    #                                            model_type=model_type, cache_dir=cache_dir)
    # imdb_test = create_imdb_dataset(root, 'test', tokenizer, max_seq_len,
    #                                 model_type=model_type, cache_dir=cache_dir)

    # squad
    root = config.data.dir.squad
    # cache_dir = osp.join(config.data.cache_dir, 'squad') if config.data.cache_dir else ''
    cache_dir = ''
    max_seq_len = config.data.max_seq_len.squad
    max_query_len = config.data.max_query_len
    trunc_stride = config.data.trunc_stride
    squad_batch_size = config.data.all_batch_size.squad
    squad_train, squad_dev = create_squad_dataset(root, 'train', tokenizer, max_seq_len, max_query_len, trunc_stride,
                                                  model_type=model_type, cache_dir=cache_dir)
    squad_test = create_squad_dataset(root, 'dev', tokenizer, max_seq_len, max_query_len, trunc_stride,
                                      model_type=model_type, cache_dir=cache_dir)

    # cnndm
    # root = config.data.dir.cnndm
    # # cache_dir = osp.join(config.data.cache_dir, 'cnndm') if config.data.cache_dir else ''
    # cache_dir = ''
    # max_src_len = config.data.max_seq_len.cnndm
    # max_tgt_len = config.data.max_tgt_len
    # cnndm_batch_size = config.data.all_batch_size.cnndm
    # cnndm_train = create_cnndm_dataset(root, 'train', tokenizer, max_src_len, max_tgt_len,
    #                                    model_type=model_type, cache_dir=cache_dir)
    # cnndm_dev = create_cnndm_dataset(root, 'val', tokenizer, max_src_len, max_tgt_len,
    #                                  model_type=model_type, cache_dir=cache_dir)
    # cnndm_test = create_cnndm_dataset(root, 'test', tokenizer, max_src_len, max_tgt_len,
    #                                   model_type=model_type, cache_dir=cache_dir)

    train_data = [squad_train]  # train_data[i]: (dataset, encoded_inputs, examples)
    dev_data = [squad_dev]
    test_data = [squad_test]
    all_batch_size = [squad_batch_size]

    # train_data = [cnndm_train]
    # dev_data = [cnndm_dev]
    # test_data = [cnndm_test]
    # all_batch_size = [cnndm_batch_size]

    data_dict = dict()
    for client_idx in range(1, config.federate.client_num + 1):
        dataloader_dict = {
            'train': {'dataloader': DataLoader(train_data[client_idx - 1][0],
                                               batch_size=all_batch_size[client_idx - 1],
                                               shuffle=config.data.shuffle,
                                               num_workers=config.data.num_workers,
                                               pin_memory=config.use_gpu),
                      'encoded': train_data[client_idx - 1][1],
                      'examples': train_data[client_idx - 1][2]},
            'val': {'dataloader': DataLoader(dev_data[client_idx - 1][0],
                                             batch_size=all_batch_size[client_idx - 1],
                                             shuffle=False,
                                             num_workers=config.data.num_workers,
                                             pin_memory=config.use_gpu),
                    'encoded': dev_data[client_idx - 1][1],
                    'examples': dev_data[client_idx - 1][2]},
            'test': {'dataloader': DataLoader(test_data[client_idx - 1][0],
                                              batch_size=all_batch_size[client_idx - 1],
                                              shuffle=False,
                                              num_workers=config.data.num_workers,
                                              pin_memory=config.use_gpu),
                     'encoded': test_data[client_idx - 1][1],
                     'examples': test_data[client_idx - 1][2]},
        }
        data_dict[client_idx] = dataloader_dict

    return data_dict, config


def call_my_data(config, **kwargs):
    if config.data.type == "mydata":
        data, modified_config = load_my_data(config, **kwargs)
        return data, modified_config
