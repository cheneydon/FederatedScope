import os
import os.path as osp
import sys

DEV_MODE = True  # simplify the federatedscope re-setup everytime we change the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from transformers.models.bert import BertTokenizerFast
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
from federatedscope.contrib.metrics.metric_builder import call_my_metric

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def extend_init_cfg(cfg):
    cfg.model.bert_type = None
    cfg.model.dec_d_ffn = None
    cfg.model.dec_dropout_prob = None
    cfg.model.num_dec_layers = None
    cfg.model.num_labels = None
    cfg.model.label_smoothing = None

    cfg.eval.sel_metric = None
    cfg.eval.is_higher_better = None
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
    cfg.data.max_tgt_len = None
    cfg.data.max_query_len = None
    cfg.data.trunc_stride = None
    cfg.data.num_labels = None
    cfg.data.all_batch_size = CN()
    cfg.data.all_batch_size.imdb = None
    cfg.data.all_batch_size.squad = None
    cfg.data.all_batch_size.cnndm = None
    cfg.data.cache_dir = None

    cfg.scheduler = CN()
    cfg.scheduler.type = None
    cfg.scheduler.warmup_ratio = None

    cfg.trainer.disp_freq = None
    cfg.trainer.val_freq = None
    cfg.trainer.grad_accum_count = None
    cfg.trainer.train_steps = None
    cfg.trainer.generator_shard_size = None
    cfg.trainer.save_dir = None

    cfg.test = CN()
    cfg.test.visible_gpus = cfg.device
    cfg.test.result_path = None
    cfg.test.beam_size = None
    cfg.test.min_length = None
    cfg.test.max_length = None
    cfg.test.block_trigram = None
    cfg.test.alpha = None
    cfg.test.recall_eval = None
    cfg.test.temp_dir = None

    cfg.optimizer.lr_enc = None
    cfg.optimizer.lr_dec = None
    cfg.optimizer.warmup_steps_enc = None
    cfg.optimizer.warmup_steps_dec = None

    return cfg


def extend_cfg_client(init_cfg, cfg_client):
    num_clients = len([k for k in cfg_client.keys() if k.startswith('client')])
    for i in range(1, num_clients + 1):
        cfg = cfg_client['client_{}'.format(i)]
        task = cfg.data.type
        cfg.data.batch_size = init_cfg.data.all_batch_size[task]
        cfg.trainer.save_dir = osp.join(init_cfg.trainer.save_dir, task)
        os.mkdir(cfg.trainer.save_dir)
    return cfg_client


def redirect_cfg_dir(cfg):
    cfg.test.result_path = cfg.outdir
    cfg.test.temp_dir = osp.join(cfg.outdir, cfg.test.temp_dir)
    cfg.trainer.save_dir = osp.join(cfg.outdir, cfg.trainer.save_dir)
    os.mkdir(cfg.test.temp_dir)
    os.mkdir(cfg.trainer.save_dir)
    return cfg


def register_all():
    register_data('mydata', call_my_data)
    register_model('mynet', call_my_net)
    register_trainer('mytrainer', call_my_trainer)
    register_metric('squad', call_my_metric)
    register_metric('rouge', call_my_metric)


if __name__ == '__main__':
    init_cfg = extend_init_cfg(global_cfg.clone())
    args = parse_args()
    init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)
    update_logger(init_cfg)
    init_cfg = redirect_cfg_dir(init_cfg)
    setup_seed(init_cfg.seed)
    register_all()

    # set up tokenizer
    bos_token, eos_token, eoq_token = '[unused0]', '[unused1]', '[unused2]'
    tokenizer = BertTokenizerFast.from_pretrained(
        init_cfg.model.bert_type,
        additional_special_tokens=[bos_token, eos_token, eoq_token],
        skip_special_tokens=True,
    )
    data, modified_cfg = get_data(config=init_cfg.clone(), tokenizer=tokenizer)
    init_cfg.merge_from_other_cfg(modified_cfg)
    init_cfg.freeze()

    # allow different settings for different clients
    cfg_client = CN.load_cfg(open(args.cfg_client, 'r'))
    cfg_client = extend_cfg_client(init_cfg, cfg_client)

    runner = FedRunner(data=data,
                       tokenizer=tokenizer,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone(),
                       config_client=cfg_client)
    _ = runner.run()
