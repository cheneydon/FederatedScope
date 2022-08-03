from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_fednlp_cfg(cfg):
    cfg.federate.num_grouped_clients = None
    cfg.federate.train_dec_out = None
    cfg.federate.train_dec_hidden = None
    cfg.federate.num_contrast = None
    cfg.federate.contrast_temp = None
    cfg.federate.load_from = None

    cfg.model.bert_type = None
    cfg.model.num_dec_layers = None
    cfg.model.num_labels = CN()
    cfg.model.num_labels.imdb = None
    cfg.model.num_labels.agnews = None
    cfg.model.num_labels.squad = None
    cfg.model.num_labels.newsqa = None
    cfg.model.num_labels.cnndm = None
    cfg.model.num_labels.msqg = None
    cfg.model.label_smoothing = None

    cfg.eval.n_best_size = None
    cfg.eval.max_answer_len = None
    cfg.eval.null_score_diff_threshold = None

    cfg.data.batch_size = None
    cfg.data.task = None
    cfg.data.max_pretrain_seq_len = None
    cfg.data.max_pretrain_tgt_len = None
    cfg.data.max_seq_len = CN()
    cfg.data.max_seq_len.imdb = None
    cfg.data.max_seq_len.agnews = None
    cfg.data.max_seq_len.squad = None
    cfg.data.max_seq_len.newsqa = None
    cfg.data.max_seq_len.cnndm = None
    cfg.data.max_seq_len.msqg = None
    cfg.data.max_tgt_len = CN()
    cfg.data.max_tgt_len.cnndm = None
    cfg.data.max_tgt_len.msqg = None
    cfg.data.max_query_len = CN()
    cfg.data.max_query_len.squad = None
    cfg.data.max_query_len.newsqa = None
    cfg.data.trunc_stride = CN()
    cfg.data.trunc_stride.squad = None
    cfg.data.trunc_stride.newsqa = None
    cfg.data.all_batch_size = CN()
    cfg.data.all_batch_size.imdb = None
    cfg.data.all_batch_size.agnews = None
    cfg.data.all_batch_size.squad = None
    cfg.data.all_batch_size.newsqa = None
    cfg.data.all_batch_size.cnndm = None
    cfg.data.all_batch_size.msqg = None
    cfg.data.collator = None
    cfg.data.pretrain_tasks = None
    cfg.data.downstream_tasks = None
    cfg.data.cache_dir = None
    cfg.data.debug = None

    cfg.aggregator = CN()
    cfg.aggregator.num_agg_groups = None
    cfg.aggregator.num_agg_topk = None
    cfg.aggregator.inside_weight = None
    cfg.aggregator.outside_weight = None

    cfg.optimizer.lr_enc = None
    cfg.optimizer.lr_dec = None
    cfg.optimizer.mlm = CN()
    cfg.optimizer.mlm.type = None
    cfg.optimizer.mlm.lr_enc = None
    cfg.optimizer.mlm.lr_dec = None
    cfg.optimizer.mlm.weight_decay = None
    cfg.optimizer.mlm.grad_clip = None
    cfg.optimizer.denoise = CN()
    cfg.optimizer.denoise.type = None
    cfg.optimizer.denoise.lr_enc = None
    cfg.optimizer.denoise.lr_dec = None
    cfg.optimizer.denoise.weight_decay = None
    cfg.optimizer.denoise.grad_clip = None

    cfg.scheduler = CN()
    cfg.scheduler.type = None
    cfg.scheduler.warmup_ratio = None
    cfg.scheduler.warmup_ratio_enc = None
    cfg.scheduler.warmup_ratio_dec = None
    cfg.scheduler.mlm = CN()
    cfg.scheduler.mlm.type = None
    cfg.scheduler.mlm.warmup_ratio_enc = None
    cfg.scheduler.mlm.warmup_ratio_dec = None
    cfg.scheduler.denoise = CN()
    cfg.scheduler.denoise.type = None
    cfg.scheduler.denoise.warmup_ratio_enc = None
    cfg.scheduler.denoise.warmup_ratio_dec = None

    cfg.trainer.disp_freq = None
    cfg.trainer.val_freq = None
    cfg.trainer.grad_accum_count = None
    cfg.trainer.train_steps = None

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

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_fednlp_cfg)


def assert_fednlp_cfg(cfg):
    pass


register_config('fednlp_cfg', extend_fednlp_cfg)
