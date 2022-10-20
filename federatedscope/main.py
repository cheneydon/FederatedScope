import os
import os.path as osp
import sys

DEV_MODE = True  # simplify the federatedscope re-setup everytime we change the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

import copy
from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg, CN
from federatedscope.core.fed_runner import FedRunner
from federatedscope.nlp.trainer.utils import setup_tokenizer

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def extend_cfg_client(init_cfg, cfg_client):
    with open(osp.join(init_cfg.outdir, 'config_client.yaml'), 'w') as outfile:
        from contextlib import redirect_stdout
        with redirect_stdout(outfile):
            tmp_cfg = copy.deepcopy(cfg_client)
            tmp_cfg.cfg_check_funcs = []
            print(tmp_cfg.dump())

    if init_cfg.federate.num_grouped_clients is None:
        num_clients = init_cfg.federate.client_num
        for i in range(1, num_clients + 1):
            cfg = cfg_client['client_{}'.format(i)]
            cfg.data.batch_size = init_cfg.data.all_batch_size[cfg.model.task]
            if init_cfg.data.debug:
                cfg.trainer.train_steps = 5
    else:
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
    cfg.test.result_path = cfg.outdir
    cfg.test.temp_dir = osp.join(cfg.outdir, 'temp')
    os.makedirs(cfg.test.temp_dir, exist_ok=True)
    if cfg.federate.save_to:
        cfg.federate.save_to = osp.join(cfg.outdir, cfg.federate.save_to)
        save_dir = cfg.federate.save_to
        os.makedirs(save_dir, exist_ok=True)

    if cfg.federate.num_grouped_clients is not None:
        if cfg.model.task == 'pretrain':
            downstream_tasks = []
            for group_id, num_clients in enumerate(cfg.federate.num_grouped_clients):
                downstream_tasks += [cfg.model.downstream_tasks[group_id]] * num_clients
            cfg.model.downstream_tasks = downstream_tasks
        elif cfg.aggregator.num_agg_topk is not None:
            num_agg_topk = []
            for group_id, num_clients in enumerate(cfg.federate.num_grouped_clients):
                if isinstance(cfg.aggregator.num_agg_topk, list):
                    num_agg_topk += [cfg.aggregator.num_agg_topk[group_id]] * num_clients
                else:
                    num_agg_topk += [cfg.aggregator.num_agg_topk] * num_clients
            cfg.aggregator.num_agg_topk = num_agg_topk

    tokenizer = setup_tokenizer(cfg)
    cfg.model.bos_token_id = tokenizer.bos_token_id
    cfg.model.eos_token_id = tokenizer.eos_token_id
    cfg.model.eoq_token_id = tokenizer.eoq_token_id
    cfg.model.pad_token_id = tokenizer.pad_token_id

    if cfg.data.debug:
        if cfg.federate.total_round_num > 5:
            cfg.federate.total_round_num = 5
        if cfg.federate.client_num > 5:
            cfg.federate.client_num = 5
        cfg.federate.save_to = ''
        if cfg.data.num_contrast is not None and cfg.data.num_contrast > 20:
            cfg.data.num_contrast = 20
        cfg.data.cache_dir = ''
        cfg.trainer.train_steps = 5

    return cfg


if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)
    update_logger(init_cfg)
    setup_seed(init_cfg.seed)
    init_cfg = extend_cfg(init_cfg)
    init_cfg.freeze()

    # allow different settings for different clients
    if args.cfg_client is None:
        cfg_client = None
    else:
        cfg_client = CN.load_cfg(open(args.cfg_client, 'r'))
        cfg_client = extend_cfg_client(init_cfg, cfg_client)

    data, _ = get_data(config=init_cfg, client_config=cfg_client)
    runner = FedRunner(data=data,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone(),
                       config_client=cfg_client)
    _ = runner.run()
