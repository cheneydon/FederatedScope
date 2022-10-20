import os.path as osp
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from federatedscope.register import register_data
from federatedscope.nlp.trainer.utils import setup_tokenizer
from federatedscope.nlp.dataset.data.imdb import create_imdb_dataset
from federatedscope.nlp.dataset.data.agnews import create_agnews_dataset
from federatedscope.nlp.dataset.data.squad import create_squad_dataset
from federatedscope.nlp.dataset.data.newsqa import create_newsqa_dataset
from federatedscope.nlp.dataset.data.cnndm import create_cnndm_dataset
from federatedscope.nlp.dataset.data.msqg import create_msqg_dataset
from federatedscope.nlp.dataset.data_collators import DataCollatorForMLM, DataCollatorForDenoisingTasks

logger = logging.getLogger(__name__)


def create_data(root, split, tokenizer, task, model_type, max_seq_len, max_query_len, trunc_stride, max_tgt_len,
                cache_dir, client_id, pretrain, debug):
    create_dataset_func = None
    if task == 'imdb':
        create_dataset_func = create_imdb_dataset
    elif task == 'agnews':
        create_dataset_func = create_agnews_dataset
    elif task == 'squad':
        create_dataset_func = create_squad_dataset
    elif task == 'newsqa':
        create_dataset_func = create_newsqa_dataset
    elif task == 'cnndm':
        create_dataset_func = create_cnndm_dataset
    elif task == 'msqg':
        create_dataset_func = create_msqg_dataset

    return create_dataset_func(root=root,
                               split=split,
                               tokenizer=tokenizer,
                               max_seq_len=max_seq_len,
                               max_query_len=max_query_len,
                               trunc_stride=trunc_stride,
                               max_src_len=max_seq_len,
                               max_tgt_len=max_tgt_len,
                               model_type=model_type,
                               cache_dir=cache_dir,
                               raw_cache_dir=cache_dir,
                               client_id=client_id,
                               pretrain=pretrain,
                               debug=debug)


def load_fednlp_data(config, client_config):
    model_type = config.model.model_type
    tokenizer = setup_tokenizer(config)
    pretrain = config.model.task == 'pretrain'
    cache_dir = config.data.cache_dir if config.data.cache_dir else ''
    debug = config.data.debug

    data_collator = None
    if pretrain:
        if config.model.pretrain_task == 'mlm':
            data_collator = DataCollatorForMLM(tokenizer=tokenizer)
        elif config.model.pretrain_task == 'denoise':
            data_collator = DataCollatorForDenoisingTasks(tokenizer=tokenizer)

    logger.info('Preprocessing dataset')
    data_dict = dict()
    for client_id in tqdm(range(1, config.federate.client_num + 1)):
        cfg_client = config if pretrain else client_config['client_{}'.format(client_id)]
        cur_task = cfg_client.model.downstream_tasks[client_id - 1] if pretrain else cfg_client.model.task
        root = osp.join(config.data.root, str(client_id))

        train_data, val_data, test_data = [create_data(root=root,
                                           split=split,
                                           tokenizer=tokenizer,
                                           task=cur_task,
                                           model_type=model_type,
                                           max_seq_len=getattr(cfg_client.data, 'max_seq_len', None),
                                           max_query_len=getattr(cfg_client.data, 'max_query_len', None),
                                           trunc_stride=getattr(cfg_client.data, 'trunc_stride', None),
                                           max_tgt_len=getattr(cfg_client.data, 'max_tgt_len', None),
                                           cache_dir=cache_dir,
                                           client_id=client_id,
                                           pretrain=pretrain,
                                           debug=debug)
                                           for split in ['train', 'val', 'test']]

        dataloader_dict = {
            'train': {'dataloader': DataLoader(dataset=train_data[0],
                                               batch_size=cfg_client.data.batch_size,
                                               shuffle=config.data.shuffle,
                                               num_workers=config.data.num_workers,
                                               collate_fn=data_collator,
                                               pin_memory=config.use_gpu),
                      'encoded': train_data[1],
                      'examples': train_data[2]},
            'val': {'dataloader': DataLoader(dataset=val_data[0],
                                             batch_size=cfg_client.data.batch_size,
                                             shuffle=False,
                                             num_workers=config.data.num_workers,
                                             collate_fn=data_collator,
                                             pin_memory=config.use_gpu),
                    'encoded': val_data[1],
                    'examples': val_data[2]},
            'test': {'dataloader': DataLoader(dataset=test_data[0],
                                              batch_size=cfg_client.data.batch_size,
                                              shuffle=False,
                                              num_workers=config.data.num_workers,
                                              collate_fn=data_collator,
                                              pin_memory=config.use_gpu),
                     'encoded': test_data[1],
                     'examples': test_data[2]},
        }
        data_dict[client_id] = dataloader_dict

    return data_dict, config


def call_fednlp_data(config, client_config):
    if config.data.type == 'fednlp_data':
        data, modified_config = load_fednlp_data(config, client_config)
        return data, modified_config


register_data('fednlp_data', call_fednlp_data)
