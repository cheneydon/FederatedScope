import logging
from federatedscope.contrib.metrics.squad import compute_squad_metrics
from federatedscope.contrib.metrics.cnndm import compute_rouge_metrics

logger = logging.getLogger(__name__)


def load_rouge_metrics(ctx, **kwargs):
    dataloader = ctx.get('{}_loader'.format(ctx.cur_data_split)).loader
    cur_step = (ctx.cur_batch_i + 1) // ctx.cfg.trainer.grad_accum_count
    results = compute_rouge_metrics(ctx.cfg.test, ctx.tokenizer, ctx.symbols, ctx.model,
                                    dataloader, cur_step, logger)
    return results


def load_squad_metric(ctx, **kwargs):
    examples = ctx.get('{}_examples'.format(ctx.cur_data_split))
    encoded_inputs = ctx.get('{}_encoded'.format(ctx.cur_data_split))
    results = ctx.get('{}_squad_results'.format(ctx.cur_data_split))
    n_best_size = ctx.cfg.eval.n_best_size
    max_answer_len = ctx.cfg.eval.max_answer_len
    null_score_diff_threshold = ctx.cfg.eval.null_score_diff_threshold

    metrics = compute_squad_metrics(
        examples, encoded_inputs, results, n_best_size, max_answer_len, null_score_diff_threshold)
    return metrics


def call_my_metric(types):
    if 'squad' in types:
        return 'squad', load_squad_metric
    if 'rouge' in types:
        return 'rouge', load_rouge_metrics
