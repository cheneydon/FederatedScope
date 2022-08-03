import logging
from federatedscope.register import register_metric
from federatedscope.nlp.metrics.generation.predictor import build_predictor

logger = logging.getLogger(__name__)


def compute_cnndm_metrics(config, tokenizer, model, testloader, step, client_id, logger):
    predictor = build_predictor(config, tokenizer, tokenizer.symbols, model, logger)
    results = predictor.translate(testloader, step, client_id=client_id)
    return results


def load_cnndm_metrics(ctx, **kwargs):
    dataloader = ctx.get('{}_loader'.format(ctx.cur_data_split)).loader
    step = (ctx.cur_batch_i + 1) // ctx.cfg.trainer.grad_accum_count
    client_id = ctx.client_id
    results = compute_cnndm_metrics(ctx.cfg.test, ctx.tokenizer, ctx.model, dataloader, step, client_id, logger)
    return results


def call_cnndm_metric(types):
    if 'cnndm' in types:
        return 'cnndm', load_cnndm_metrics


register_metric('cnndm', call_cnndm_metric)
