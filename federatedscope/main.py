import os
import sys

DEV_MODE = True  # simplify the federatedscope re-setup everytime we change the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg, CN
from federatedscope.core.fed_runner import FedRunner
from federatedscope.register import register_data, register_model, register_trainer, register_metric
from federatedscope.contrib.data.data_builder import call_my_data
from federatedscope.contrib.models.model import call_my_net
from federatedscope.contrib.trainers.trainer import call_my_trainer
from federatedscope.contrib.metrics.squad import call_squad_metric

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']


def extend_cfg(cfg):
    cfg.model.bert_type = None
    cfg.model.num_dec_layers = None
    cfg.model.num_labels = None

    cfg.eval.sel_metric = None
    cfg.eval.n_best_size = None
    cfg.eval.max_answer_len = None
    cfg.eval.null_score_diff_threshold = None

    cfg.data.dir = CN()
    cfg.data.dir.imdb = None
    cfg.data.dir.squad = None
    cfg.data.dir.cnndm = None
    cfg.data.max_seq_len = CN()
    cfg.data.max_seq_len.imdb = None
    cfg.data.max_seq_len.squad = None
    cfg.data.max_seq_len.cnndm = None
    cfg.data.max_query_len = CN()
    cfg.data.max_query_len.squad = None
    cfg.data.trunc_stride = CN()
    cfg.data.trunc_stride.squad = None
    cfg.data.num_labels = None

    cfg.scheduler = CN()
    cfg.scheduler.type = None
    cfg.scheduler.warmup_ratio = None

    cfg.trainer.disp_freq = None
    cfg.trainer.val_freq = None

    return cfg


def register_all():
    register_data('mydata', call_my_data)
    register_model('mynet', call_my_net)
    register_trainer('mytrainer', call_my_trainer)
    register_metric('squad', call_squad_metric)


if __name__ == '__main__':
    init_cfg = extend_cfg(global_cfg.clone())
    args = parse_args()
    init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)

    update_logger(init_cfg)
    setup_seed(init_cfg.seed)
    register_all()

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global cfg object
    data, modified_cfg = get_data(config=init_cfg.clone())
    init_cfg.merge_from_other_cfg(modified_cfg)
    init_cfg.freeze()

    # allow different settings for different clients
    cfg_client = CN.load_cfg(open(args.cfg_client, 'r'))

    runner = FedRunner(data=data,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone(),
                       config_client=cfg_client)
    _ = runner.run()
