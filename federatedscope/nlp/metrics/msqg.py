import os.path as osp
import logging
from federatedscope.register import register_metric
from federatedscope.nlp.metrics.generation.predictor import build_predictor
from federatedscope.nlp.metrics.generation.eval import eval

logger = logging.getLogger(__name__)


def compute_msqg_metrics(config, tokenizer, model, testloader, step, client_id, logger):
    predictor = build_predictor(config, tokenizer, tokenizer.symbols, model, logger)
    rouge_results = predictor.translate(testloader, step, client_id=client_id)

    pred_file = osp.join(config.result_path, 'pred', '{}.txt'.format(client_id))
    src_file = osp.join(config.result_path, 'src', '{}.txt'.format(client_id))
    tgt_file = osp.join(config.result_path, 'tgt', '{}.txt'.format(client_id))
    qg_results = eval(pred_file, src_file, tgt_file)  # bleu & meteor

    results = rouge_results
    results.update(qg_results)
    results = {k: v for k, v in results.items() if k in ('rouge_l_f_score', 'Bleu_4', 'METEOR')}
    return results


def load_msqg_metrics(ctx, **kwargs):
    dataloader = ctx.get('{}_loader'.format(ctx.cur_data_split)).loader
    step = (ctx.cur_batch_i + 1) // ctx.cfg.trainer.grad_accum_count
    client_id = ctx.client_id
    results = compute_msqg_metrics(ctx.cfg.test, ctx.tokenizer, ctx.model, dataloader, step, client_id, logger)
    return results


def call_msqg_metric(types):
    if 'msqg' in types:
        return 'msqg', load_msqg_metrics


register_metric('msqg', call_msqg_metric)
