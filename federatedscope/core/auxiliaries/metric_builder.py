import federatedscope.register as register
from federatedscope.contrib.metrics import *
from federatedscope.nlp.metrics import *


def get_metric(types):
    metrics = dict()
    for func in register.metric_dict.values():
        res = func(types)
        if res is not None:
            name, metric = res
            metrics[name] = metric
    return metrics
