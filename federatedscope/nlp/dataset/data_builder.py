from torch.utils.data import DataLoader
from federatedscope.register import register_data
from federatedscope.nlp.dataset.sts import create_sts_dataset
from federatedscope.nlp.dataset.imdb import create_imdb_dataset
from federatedscope.nlp.dataset.squad import create_squad_dataset


def load_my_data(config, tokenizer, **kwargs):
    model_type = config.model.bert_type

    # sts-b
    root = config.data.dir.sts
    max_seq_len = config.data.max_seq_len.sts
    sts_batch_size = config.data.all_batch_size.sts
    sts_train, sts_dev = create_sts_dataset(root, 'train', tokenizer, max_seq_len,
                                            model_type=model_type, cache_dir=config.data.cache_dir)
    sts_test = create_sts_dataset(root, 'dev', tokenizer, max_seq_len, model_type=model_type,
                                  cache_dir=config.data.cache_dir)

    # imdb
    root = config.data.dir.imdb
    max_seq_len = config.data.max_seq_len.imdb
    imdb_batch_size = config.data.all_batch_size.imdb
    imdb_train, imdb_dev = create_imdb_dataset(root, 'train', tokenizer, max_seq_len,
                                               model_type=model_type, cache_dir=config.data.cache_dir)
    imdb_test = create_imdb_dataset(root, 'test', tokenizer, max_seq_len, model_type=model_type,
                                    cache_dir=config.data.cache_dir)

    # squad
    root = config.data.dir.squad
    max_seq_len = config.data.max_seq_len.squad
    max_query_len = config.data.max_query_len
    trunc_stride = config.data.trunc_stride
    squad_batch_size = config.data.all_batch_size.squad
    squad_train, squad_dev = create_squad_dataset(root, 'train', tokenizer, max_seq_len, max_query_len,
                                                  trunc_stride, model_type=model_type, cache_dir=config.data.cache_dir)
    squad_test = create_squad_dataset(root, 'dev', tokenizer, max_seq_len, max_query_len,
                                      trunc_stride, model_type=model_type, cache_dir=config.data.cache_dir)

    # # cnndm
    # root = config.data.dir.cnndm
    # max_src_len = config.data.max_seq_len.cnndm
    # max_tgt_len = config.data.max_tgt_len
    # cnndm_batch_size = config.data.all_batch_size.cnndm
    # cnndm_train = create_cnndm_dataset(root, 'train', tokenizer, max_src_len, max_tgt_len,
    #                                    model_type=model_type, raw_cache_dir=config.data.cache_dir)
    # cnndm_dev = create_cnndm_dataset(root, 'val', tokenizer, max_src_len, max_tgt_len,
    #                                  model_type=model_type, raw_cache_dir=config.data.cache_dir)
    # cnndm_test = create_cnndm_dataset(root, 'test', tokenizer, max_src_len, max_tgt_len,
    #                                   model_type=model_type, raw_cache_dir=config.data.cache_dir)

    # train_data = [imdb_train, squad_train, cnndm_train]  # train_data[i]: (dataset, encoded_inputs, examples)
    # dev_data = [imdb_dev, squad_dev, cnndm_dev]
    # test_data = [imdb_test, squad_test, cnndm_test]
    # all_batch_size = [imdb_batch_size, squad_batch_size, cnndm_batch_size]

    train_data = [sts_train, imdb_train, squad_train]
    dev_data = [sts_dev, imdb_dev, squad_dev]
    test_data = [sts_test, imdb_test, squad_test]
    all_batch_size = [sts_batch_size, imdb_batch_size, squad_batch_size]

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
    if config.data.type == 'mydata':
        data, modified_config = load_my_data(config, **kwargs)
        return data, modified_config


register_data('mydata', call_my_data)
