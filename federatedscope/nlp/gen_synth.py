import os
import torch
import copy
import random
import logging
import json
import numpy as np
from tqdm import tqdm
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.configs.config import global_cfg, CN
from federatedscope.nlp.trainer.utils import setup_tokenizer
from federatedscope.nlp.model.pfednlp_model import PFedNLPModel
from federatedscope.nlp.dataset.pfednlp_data import load_pfednlp_data


def extend_cfg_client(init_cfg, cfg_client):
    num_grouped_clients = init_cfg.federate.num_grouped_clients
    client_start_id = 1
    for group_id, num_clients in enumerate(num_grouped_clients):
        group_cfg = cfg_client['client_group_{}'.format(group_id + 1)]
        if init_cfg.data.debug:
            group_cfg.trainer.train_steps = 5
        for client_id in range(client_start_id, client_start_id + num_clients):
            cfg_client['client_{}'.format(client_id)] = group_cfg
        client_start_id += num_clients

    return cfg_client


def extend_cfg(cfg):
    cfg.outdir = out_dir
    cfg.data.cache_dir = cache_dir
    cfg.federate.load_from = pretrain_dir
    cfg.data.debug = DEBUG
    cfg.data.shuffle = False
    if cfg.federate.num_grouped_clients is not None and cfg.model.task == 'pretrain':
        downstream_tasks = []
        num_grouped_clients = cfg.federate.num_grouped_clients
        for group_id, num_clients in enumerate(num_grouped_clients):
            downstream_tasks += [cfg.model.downstream_tasks[group_id]] * num_clients
        cfg.model.downstream_tasks = downstream_tasks

    tokenizer = setup_tokenizer(cfg)
    cfg.model.bos_token_id = tokenizer.bos_token_id
    cfg.model.eos_token_id = tokenizer.eos_token_id
    cfg.model.eoq_token_id = tokenizer.eoq_token_id
    cfg.model.pad_token_id = tokenizer.pad_token_id

    if cfg.data.debug:
        if cfg.federate.total_round_num > 5:
            cfg.federate.total_round_num = 5
        if cfg.federate.client_num > NUM_CLIENT:
            cfg.federate.client_num = NUM_CLIENT
            cfg.aggregator.num_agg_groups = 1
        cfg.federate.save_to = ''
        cfg.data.cache_dir = ''
        cfg.trainer.train_steps = 5

    return cfg


def get_model(init_cfg, cfg_client):
    models = {}
    for client_id in range(1, NUM_CLIENT + 1):
        cfg = init_cfg.clone()
        cfg.merge_from_other_cfg(cfg_client.get('client_{}'.format(client_id)))
        model = PFedNLPModel(cfg.model)

        load_path = cfg.federate.load_from
        global_dir = os.path.join(load_path, 'global')
        client_dir = os.path.join(load_path, 'client')
        global_ckpt_path = os.path.join(global_dir, 'global_model_{}.pt'.format(client_id))
        client_ckpt_path = os.path.join(client_dir, 'client_model_{}.pt'.format(client_id))
        if os.path.exists(global_ckpt_path):
            model_ckpt = model.state_dict()
            logger.info('Loading model from \'{}\''.format(global_ckpt_path))
            global_ckpt = torch.load(global_ckpt_path, map_location='cpu')['model']
            model_ckpt.update(global_ckpt)
            if os.path.exists(client_ckpt_path):
                logger.info('Updating model from \'{}\''.format(client_ckpt_path))
                client_ckpt = torch.load(client_ckpt_path, map_location='cpu')['model']
                model_ckpt.update(client_ckpt)
            model.load_state_dict(model_ckpt)
        else:
            raise RuntimeError('Checkpoint NOT found in \'{}\''.format(global_ckpt_path))
        models[client_id] = model
    return models


def get_avg_mlm_head(models):
    all_params = copy.deepcopy([models[k].lm_head.state_dict() for k in range(1, NUM_CLIENT + 1)])
    avg_param = all_params[0]
    for k in avg_param:
        for i in range(len(all_params)):
            local_param = all_params[i][k].float()
            if i == 0:
                avg_param[k] = local_param / len(all_params)
            else:
                avg_param[k] += local_param / len(all_params)
    avg_model = copy.deepcopy(models[1].lm_head)
    avg_model.load_state_dict(avg_param)
    return avg_model


if __name__ == '__main__':
    DEBUG = False #True
    DEVICE_ID = '7'
    NUM_CLIENT = 18
    BATCH_SIZE = 32
    FEAT_DIM = 128
    PRIM_RATIO = 0.5

    config_dir = os.path.dirname(os.path.realpath(__file__))
    pretrain_dir = os.path.join(os.path.dirname(config_dir), 'exp/pfednlp_pt/v5/bert2bert/200_50/group_5/pretrain/ckpt/')
    cache_dir = os.path.join(os.path.dirname(config_dir), 'cache/v5/')
    out_dir = os.path.join(os.path.dirname(config_dir), 'exp/synthetic/')
    device = torch.device('cuda:{}'.format(DEVICE_ID))

    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(os.path.join(config_dir, 'configs/pfednlp/config_pfednlp.yaml'))
    init_cfg = extend_cfg(init_cfg)
    update_logger(init_cfg)
    logger = logging.getLogger('federatedscope')

    setup_seed(init_cfg.seed)
    cfg_client = CN.load_cfg(open(os.path.join(config_dir, 'configs/pfednlp/config_client_pfednlp.yaml'), 'r'))
    cfg_client = extend_cfg_client(init_cfg, cfg_client)
    data, _ = load_pfednlp_data(init_cfg, cfg_client)
    models = get_model(init_cfg, cfg_client)

    logger.info('Generating encoder hidden states')
    max_sz, max_len = 1e8, 0
    for client_id in range(1, NUM_CLIENT + 1):
        dataset = data[client_id]['train']['dataloader'].dataset
        max_sz = min(max_sz, len(dataset))
        max_len = max(max_len, len(dataset[0]['token_ids']))
    enc_hiddens = np.memmap(filename=os.path.join(init_cfg.outdir, 'feature.memmap'),
                            shape=(NUM_CLIENT, max_sz, max_len, FEAT_DIM),
                            mode='w+',
                            dtype=np.float32)

    for client_id in range(1, NUM_CLIENT + 1):
        dataloader = data[client_id]['train']['dataloader']
        model = models[client_id]
        model.eval()
        model.to(device)
        enc_hid = []
        for batch_i, data_batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            token_ids = data_batch['token_ids']
            token_type_ids = data_batch['token_type_ids']
            attention_mask = data_batch['attention_mask']

            enc_out = model.model.encoder(
                input_ids=token_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )
            enc_hid.append(enc_out.last_hidden_state.detach().cpu())
        enc_hid = torch.cat(enc_hid)
        if enc_hid.size(1) < max_len:
            enc_hid = torch.cat([enc_hid, torch.zeros(enc_hid.size(0), max_len - enc_hid.size(1), FEAT_DIM)], dim=1)
        enc_hiddens[client_id - 1] = enc_hid[:max_sz]
        model.to(torch.device('cpu'))

    all_hids = torch.from_numpy(enc_hiddens)
    prim_indices = [random.randint(0, len(all_hids) - 1) for _ in range(len(all_hids[0]))]
    all_ratios = torch.ones(len(all_hids), len(all_hids[0]))
    all_ratios *= (1 - PRIM_RATIO) / (len(all_hids) - 1)
    for i, pi in enumerate(prim_indices):
        all_ratios[pi, i] = PRIM_RATIO
    avg_hids = (all_hids * all_ratios[:, :, None, None]).sum(0)

    logger.info('Generating input tokens')
    mlm_head = get_avg_mlm_head(models).to(device)
    with torch.no_grad():
        pred_toks = torch.cat([
            mlm_head(avg_hids[i: i + BATCH_SIZE].to(device)).detach().cpu().argmax(dim=-1)
            for i in tqdm(range(0, avg_hids.size(0), BATCH_SIZE))])

    if cache_dir:
        cache_dir = os.path.join(cache_dir, 'synthetic')
        logger.info('Saving synthetic data to \'{}\''.format(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        saved_feats = np.memmap(filename=os.path.join(cache_dir, 'feature_{}.memmap'.format(PRIM_RATIO)),
                                shape=avg_hids.size(),
                                mode='w+',
                                dtype=np.float32)
        saved_toks = np.memmap(filename=os.path.join(cache_dir, 'token_{}.memmap'.format(PRIM_RATIO)),
                               shape=pred_toks.size(),
                               mode='w+',
                               dtype=np.int64)
        for i in range(len(avg_hids)):
            saved_feats[i] = avg_hids[i]
            saved_toks[i] = pred_toks[i]

        shapes = {'feature': avg_hids.size(), 'token': pred_toks.size()}
        with open(os.path.join(cache_dir, 'shapes.json'), 'w') as f:
            json.dump(shapes, f)
