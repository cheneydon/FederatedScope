import logging
from federatedscope.register import register_metric
from federatedscope.nlp.metrics.generation.utils import test_rouge

logger = logging.getLogger(__name__)


def load_cnndm_metrics(ctx, **kwargs):
    rouges = test_rouge(ctx.cfg.test.temp_dir, ctx.pred_path, ctx.tgt_path)
    results = {k: v for k, v in rouges.items()
               if k in {'rouge_1_f_score', 'rouge_2_f_score', 'rouge_l_f_score'}}
    return results


def call_cnndm_metric(types):
    if 'cnndm' in types:
        return 'cnndm', load_cnndm_metrics


register_metric('cnndm', call_cnndm_metric)
