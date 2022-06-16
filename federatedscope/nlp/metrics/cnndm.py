from federatedscope.nlp.metrics.generation.predictor import build_predictor


def compute_cnndm_metrics(config, tokenizer, symbols, model, testloader, step, logger):
    predictor = build_predictor(config, tokenizer, symbols, model, logger)
    results = predictor.translate(testloader, step)
    return results
