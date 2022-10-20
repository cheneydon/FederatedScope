from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_fednlp_cfg(cfg):
    cfg.federate.num_grouped_clients = None
    cfg.federate.load_from = None

    cfg.model.model_type = None
    cfg.model.task = None
    cfg.model.pretrain_task = None
    cfg.model.pretrain_tasks = None
    cfg.model.downstream_tasks = None
    cfg.model.num_labels = None
    cfg.model.max_length = None
    cfg.model.min_length = None
    cfg.model.no_repeat_ngram_size = None
    cfg.model.length_penalty = None
    cfg.model.num_beams = None
    cfg.model.label_smoothing = None
    cfg.model.n_best_size = None
    cfg.model.max_answer_len = None
    cfg.model.null_score_diff_threshold = None
    cfg.model.bos_token = None
    cfg.model.eos_token = None
    cfg.model.eoq_token = None
    cfg.model.train_contrast = None
    cfg.model.contrast_topk = None
    cfg.model.contrast_temp = None

    cfg.data.batch_size = None
    cfg.data.contrast_batch_size = None
    cfg.data.max_seq_len = None
    cfg.data.max_tgt_len = None
    cfg.data.max_query_len = None
    cfg.data.trunc_stride = None
    cfg.data.cache_dir = None
    cfg.data.num_contrast = None
    cfg.data.debug = None

    cfg.aggregator = CN()
    cfg.aggregator.num_agg_groups = None
    cfg.aggregator.num_agg_topk = None
    cfg.aggregator.inside_weight = None
    cfg.aggregator.outside_weight = None
    cfg.aggregator.proto_weight = None
    cfg.aggregator.synth_ratio = None

    cfg.optimizer.lr = None
    cfg.optimizer.mlm = CN()
    cfg.optimizer.mlm.type = None
    cfg.optimizer.mlm.lr = None
    cfg.optimizer.mlm.weight_decay = None
    cfg.optimizer.mlm.grad_clip = None
    cfg.optimizer.denoise = CN()
    cfg.optimizer.denoise.type = None
    cfg.optimizer.denoise.lr = None
    cfg.optimizer.denoise.weight_decay = None
    cfg.optimizer.denoise.grad_clip = None

    cfg.scheduler = CN()
    cfg.scheduler.type = None
    cfg.scheduler.warmup_ratio = None
    cfg.scheduler.warmup_ratio = None
    cfg.scheduler.mlm = CN()
    cfg.scheduler.mlm.type = None
    cfg.scheduler.mlm.warmup_ratio = None
    cfg.scheduler.denoise = CN()
    cfg.scheduler.denoise.type = None
    cfg.scheduler.denoise.warmup_ratio = None

    cfg.trainer.disp_freq = None
    cfg.trainer.val_freq = None
    cfg.trainer.grad_accum_count = None
    cfg.trainer.train_steps = None

    cfg.test = CN()
    cfg.test.result_path = None
    cfg.test.temp_dir = None

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_fednlp_cfg)


def assert_fednlp_cfg(cfg):
    pass


register_config('fednlp_cfg', extend_fednlp_cfg)
