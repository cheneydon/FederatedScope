import logging
from federatedscope.register import register_metric
from federatedscope.nlp.metrics.sts import compute_sts_metrics
from federatedscope.nlp.metrics.squad import compute_squad_metrics
from federatedscope.nlp.metrics.cnndm import compute_cnndm_metrics

logger = logging.getLogger(__name__)


def load_sts_metrics(y_true, y_prob, **kwargs):
    results = compute_sts_metrics(y_prob.squeeze(), y_true.squeeze())
    return results


def load_squad_metrics(ctx, **kwargs):
    examples = ctx.get('{}_examples'.format(ctx.cur_data_split))
    encoded_inputs = ctx.get('{}_encoded'.format(ctx.cur_data_split))
    results = ctx.get('{}_squad_results'.format(ctx.cur_data_split))
    n_best_size = ctx.cfg.eval.n_best_size
    max_answer_len = ctx.cfg.eval.max_answer_len
    null_score_diff_threshold = ctx.cfg.eval.null_score_diff_threshold

    metrics = compute_squad_metrics(
        examples, encoded_inputs, results, n_best_size, max_answer_len, null_score_diff_threshold)
    return metrics


def load_cnndm_metrics(ctx, **kwargs):
    dataloader = ctx.get('{}_loader'.format(ctx.cur_data_split)).loader
    cur_step = (ctx.cur_batch_i + 1) // ctx.cfg.trainer.grad_accum_count
    results = compute_cnndm_metrics(ctx.cfg.test, ctx.tokenizer, ctx.tokenizer.symbols, ctx.model,
                                    dataloader, cur_step, logger)
    return results


def call_sts_metric(types):
    if 'sts' in types:
        return 'sts', load_sts_metrics

def call_squad_metric(types):
    if 'squad' in types:
        return 'squad', load_squad_metrics

def call_rouge_metric(types):
    if 'rouge' in types:
        return 'rouge', load_cnndm_metrics


register_metric('sts', call_sts_metric)
register_metric('squad', call_squad_metric)
register_metric('rouge', call_rouge_metric)
