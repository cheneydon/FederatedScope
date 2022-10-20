import logging

from federatedscope.core.configs import constants

logger = logging.getLogger(__name__)


def get_aggregator(method, model=None, device=None, online=False, config=None):
    if config.backend == 'tensorflow':
        from federatedscope.cross_backends import FedAvgAggregator
        return FedAvgAggregator(model=model, device=device)
    else:
        from federatedscope.core.aggregator import ClientsAvgAggregator, OnlineClientsAvgAggregator, ServerClientsInterpolateAggregator, FedOptAggregator, NoCommunicationAggregator

    if config.trainer.type == 'pfednlp_trainer':
        from federatedscope.core.aggregator import PFedNLPAggregator
        return PFedNLPAggregator(model=model, config=config, device=device)

    if config.trainer.type == 'pfednlp_contrast_trainer':
        from federatedscope.core.aggregator import PFedNLPContrastAggregator
        return PFedNLPContrastAggregator(model=model, config=config, device=device)

    if config.trainer.type == 'spfl_trainer':
        from federatedscope.core.aggregator import SPFLAggregator
        return SPFLAggregator(model=model, config=config, device=device)

    if config.trainer.type == 'percfl_trainer':
        from federatedscope.core.aggregator import PerCFLAggregator
        return PerCFLAggregator(model=model, config=config, device=device)

    if method.lower() in constants.AGGREGATOR_TYPE:
        aggregator_type = constants.AGGREGATOR_TYPE[method.lower()]
    else:
        aggregator_type = "clients_avg"
        logger.warning(
            'Aggregator for method {} is not implemented. Will use default one'
            .format(method))

    if config.fedopt.use or aggregator_type == 'fedopt':
        return FedOptAggregator(config=config, model=model, device=device)
    elif aggregator_type == 'clients_avg':
        if online:
            return OnlineClientsAvgAggregator(
                model=model,
                device=device,
                src_device=device
                if config.federate.share_local_model else 'cpu')
        else:
            return ClientsAvgAggregator(model=model, device=device)
    elif aggregator_type == 'server_clients_interpolation':
        return ServerClientsInterpolateAggregator(
            model=model, device=device, beta=config.personalization.beta)
    elif aggregator_type == 'no_communication':
        return NoCommunicationAggregator()
    else:
        raise NotImplementedError(
            "Aggregator {} is not implemented.".format(aggregator_type))
