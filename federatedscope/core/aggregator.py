import os
import re
import math
import copy
import random
import torch
import numpy as np
import logging
from abc import ABC, abstractmethod
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from transformers.models.bert import BertForPreTraining
from federatedscope.nlp.trainer.utils import ContrastiveMonitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.configs.config import global_cfg

logger = logging.getLogger(__name__)


class Aggregator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, agg_info):
        pass


class ClientsAvgAggregator(Aggregator):
    """Implementation of vanilla FedAvg refer to `Communication-efficient learning of deep networks from decentralized data` [McMahan et al., 2017]
        (http://proceedings.mlr.press/v54/mcmahan17a.html)
    """
    def __init__(self, model=None, device='cpu'):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and global_cfg.federate.use_ss) else None
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)

        return avg_model

    def update(self, model_parameters):
        '''
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        '''
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

    def _para_weighted_avg(self, models, recover_fun=None):
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = models[0]
        for key in avg_model:
            for i in range(len(models)):
                local_sample_size, local_model = models[i]

                if global_cfg.federate.ignore_weight:
                    weight = 1.0 / len(models)
                elif global_cfg.federate.use_ss:
                    # When using secret sharing, what the server receives are sample_size * model_para
                    weight = 1.0
                else:
                    weight = local_sample_size / training_set_size

                if not global_cfg.federate.use_ss:
                    if isinstance(local_model[key], torch.Tensor):
                        local_model[key] = local_model[key].float()
                    else:
                        local_model[key] = torch.FloatTensor(local_model[key])

                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

            if global_cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                # When using secret sharing, what the server receives are sample_size * model_para
                avg_model[key] /= training_set_size
                avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model


class NoCommunicationAggregator(Aggregator):
    """"Clients do not communicate. Each client work locally
    """
    def aggregate(self, agg_info):
        # do nothing
        return {}


class OnlineClientsAvgAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, device='cpu', src_device='cpu'):
        super(OnlineClientsAvgAggregator, self).__init__(model, device)
        self.src_device = src_device

    def reset(self):
        self.maintained = self.model.state_dict()
        for key in self.maintained:
            self.maintained[key].data = torch.zeros_like(
                self.maintained[key], device=self.src_device)
        self.cnt = 0

    def inc(self, content):
        if isinstance(content, tuple):
            sample_size, model_params = content
            for key in self.maintained:
                # if model_params[key].device != self.maintained[key].device:
                #    model_params[key].to(self.maintained[key].device)
                self.maintained[key] = (self.cnt * self.maintained[key] +
                                        sample_size * model_params[key]) / (
                                            self.cnt + sample_size)
            self.cnt += sample_size
        else:
            raise TypeError(
                "{} is not a tuple (sample_size, model_para)".format(content))

    def aggregate(self, agg_info):
        return self.maintained


class ServerClientsInterpolateAggregator(ClientsAvgAggregator):
    """"
        # conduct aggregation by interpolating global model from server and local models from clients
    """
    def __init__(self, model=None, device='cpu', beta=1.0):
        super(ServerClientsInterpolateAggregator, self).__init__(model, device)
        self.beta = beta  # the weight for local models used in interpolation

    def aggregate(self, agg_info):
        models = agg_info["client_feedback"]
        global_model = self.model
        elem_each_client = next(iter(models))
        assert len(elem_each_client) == 2, f"Require (sample_size, model_para) tuple for each client, " \
                                           f"i.e., len=2, but got len={len(elem_each_client)}"
        avg_model_by_clients = self._para_weighted_avg(models)
        global_local_models = [((1 - self.beta), global_model.state_dict()),
                               (self.beta, avg_model_by_clients)]

        avg_model_by_interpolate = self._para_weighted_avg(global_local_models)
        return avg_model_by_interpolate


class FedOptAggregator(ClientsAvgAggregator):
    """Implementation of FedOpt refer to `Adaptive Federated Optimization` [Reddi et al., 2021]
        (https://openreview.net/forum?id=LkFG3lB13U5)

    """
    def __init__(self, config, model, device='cpu'):
        super(FedOptAggregator, self).__init__(model, device)
        self.cfg = config
        self.model = model
        self.device = device
        self.optimizer = get_optimizer(type=config.fedopt.type_optimizer,
                                       model=self.model,
                                       lr=config.fedopt.lr_server)

    def aggregate(self, agg_info):
        new_model = super().aggregate(agg_info)

        model = self.model.cpu().state_dict()
        with torch.no_grad():
            grads = {key: model[key] - new_model[key] for key in new_model}

        self.optimizer.zero_grad()
        for key, p in self.model.named_parameters():
            if key in new_model.keys():
                p.grad = grads[key]
        self.optimizer.step()

        return self.model.state_dict()


class PFedNLPAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, config=None, device='cpu'):
        super().__init__(model, device)
        self.config = config
        self.client_num = config.federate.client_num
        self.task = config.data.task
        self.pretrain_tasks = config.data.pretrain_tasks
        self.num_agg_groups = config.aggregator.num_agg_groups
        self.num_agg_topk = config.aggregator.num_agg_topk
        self.inside_weight = config.aggregator.inside_weight
        self.outside_weight = config.aggregator.outside_weight
        self.models = []
        self.neighbors = {}
        self.client_id2group = [None for _ in range(self.client_num)]
        self.client_id2topk = [[] for _ in range(self.client_num)]

    def update_models(self, models):
        self.models = models

    def update_neighbors(self, neighbors):
        self.neighbors = neighbors

    def aggregate(self, agg_info):
        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if ('recover_fun' in agg_info and global_cfg.federate.use_ss) else None
        avg_models, tasks = self._para_weighted_avg(models, recover_fun=recover_fun)
        return avg_models, tasks

    def update(self, model_parameters):
        for i, param in enumerate(model_parameters):
            self.models[i].load_state_dict(param, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.models is not None

        path = os.path.join(path, 'global')
        os.makedirs(path, exist_ok=True)
        neighbor_ids = list(self.neighbors.keys())
        for i, model in enumerate(self.models):
            ckpt = {'cur_round': cur_round, 'model': model.state_dict()}
            torch.save(ckpt, os.path.join(path, 'global_model_{}.pt'.format(neighbor_ids[i])))

    def load_model(self, path):
        if getattr(self, 'models', None):
            round = None
            global_dir = os.path.join(path, 'global')
            client_dir = os.path.join(path, 'client')
            neighbor_ids = sorted([int(re.search(r'model_(\d+).pt', x).groups()[0]) for x in os.listdir(global_dir)])
            assert len(neighbor_ids) == len(self.models)
            for i, model in enumerate(self.models):
                cur_global_path = os.path.join(global_dir, 'global_model_{}.pt'.format(neighbor_ids[i]))
                cur_client_path = os.path.join(client_dir, 'client_model_{}.pt'.format(neighbor_ids[i]))
                if os.path.exists(cur_global_path):
                    model_ckpt = model.state_dict()
                    logger.info('Loading model from \'{}\''.format(cur_global_path))
                    global_ckpt = torch.load(cur_global_path, map_location=self.device)
                    model_ckpt.update(global_ckpt['model'])
                    if os.path.exists(cur_client_path):
                        logger.info('Updating model from \'{}\''.format(cur_client_path))
                        client_ckpt = torch.load(cur_client_path, map_location=self.device)
                        model_ckpt.update(client_ckpt['model'])
                    self.models[i].load_state_dict(model_ckpt)
                    round = global_ckpt['cur_round']
                else:
                    raise ValueError("The file {} does NOT exist".format(cur_global_path))
            return round

    def _compute_client_groups(self, models):
        tasks = [None for _ in range(self.client_num)]
        if self.task == 'pretrain':
            task_groups = [random.randint(0, len(self.pretrain_tasks) - 1) for _ in range(self.num_agg_groups)]
            grads = torch.stack([model['model_grads'] for model in models])
            clustering = AgglomerativeClustering(n_clusters=self.num_agg_groups,
                                                 affinity='cosine',
                                                 linkage='average').fit(grads)
            self.client_id2group = clustering.labels_
            task_ids = [task_groups[self.client_id2group[i]] for i in range(self.client_num)]
            tasks = [self.pretrain_tasks[i] for i in task_ids]
        else:
            grads = torch.stack([model['model_grads'] for model in models])
            distances = cosine_distances(grads, grads)
            self.client_id2topk = np.argsort(distances, axis=-1)[:, :self.num_agg_topk].tolist()
        return tasks

    def _avg_params(self, models, client_adj_norm):
        model_params = [copy.deepcopy(model['model_para']) for model in models]
        for k in model_params[0]:
            if isinstance(model_params[0][k], torch.FloatTensor):
                cur_params = torch.stack([param[k] for param in model_params])
                avg_params = None
                if cur_params.ndim == 4:
                    avg_params = torch.einsum('ij,jklm->iklm', client_adj_norm, cur_params)
                elif cur_params.ndim == 3:
                    avg_params = torch.einsum('ij,jkl->ikl', client_adj_norm, cur_params)
                elif cur_params.ndim == 2:
                    avg_params = torch.einsum('ij,jk->ik', client_adj_norm, cur_params)
                for i in range(len(model_params)):
                    model_params[i][k] = avg_params[i]
        return model_params

    def _para_weighted_avg(self, models, recover_fun=None):
        if self.config.federate.method in ['local', 'global']:
            model_params = [model['model_para'] for model in models]
            tasks = [None for _ in range(self.client_num)]
            return model_params, tasks

        if self.task == 'pretrain':
            # generate param weight matrix
            client_adj = torch.zeros(self.client_num, self.client_num)
            for i in range(self.client_num):
                for j in range(self.client_num):
                    if self.client_id2group[i] == self.client_id2group[j]:
                        client_adj[i][j] = models[j]['sample_size'] * self.inside_weight
                    else:
                        client_adj[i][j] = models[j]['sample_size'] * self.outside_weight
            client_adj_norm = client_adj / client_adj.sum(dim=-1, keepdim=True)

            # aggregate model params
            model_params = self._avg_params(models, client_adj_norm)
            # generate group task and self.client_id2group
            tasks = self._compute_client_groups(models)
            logger.info('client_id2group: {}'.format({k + 1: v for k, v in enumerate(self.client_id2group)}))
            return model_params, tasks

        else:
            # generate self.client_id2topk and param weight matrix
            tasks = self._compute_client_groups(models)
            logger.info('client_id2topk: {}'.format({k + 1: v for k, v in enumerate(self.client_id2topk)}))
            client_adj = torch.zeros(self.client_num, self.client_num)
            for i in range(self.client_num):
                for j in range(self.client_num):
                    if j in self.client_id2topk[i]:
                        client_adj[i][j] = models[j]['sample_size'] * self.inside_weight
                    else:
                        client_adj[i][j] = models[j]['sample_size'] * self.outside_weight
            client_adj_norm = client_adj / client_adj.sum(dim=-1, keepdim=True)

            # aggregate model params
            model_params = self._avg_params(models, client_adj_norm)
            return model_params, tasks


class PFedNLPContrastAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, config=None, device='cpu'):
        super().__init__(model, device)
        self.config = config
        self.client_num = config.federate.client_num
        self.task = config.data.task
        self.pretrain_tasks = config.data.pretrain_tasks
        self.num_agg_groups = config.aggregator.num_agg_groups
        self.num_agg_topk = config.aggregator.num_agg_topk
        self.inside_weight = config.aggregator.inside_weight
        self.outside_weight = config.aggregator.outside_weight
        self.models = []
        self.neighbors = {}
        self.client_id2group = [None for _ in range(self.client_num)]
        self.client_id2topk = [[] for _ in range(self.client_num)]
        self.contrast_monitor = ContrastiveMonitor()
        self.encoder = BertForPreTraining.from_pretrained(config.model.bert_type).to(config.device)
        self.encoder.eval()

    def aggregate(self, agg_info):
        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if ('recover_fun' in agg_info and global_cfg.federate.use_ss) else None
        avg_models, tasks = self._para_weighted_avg(models, recover_fun=recover_fun)
        return avg_models, tasks

    def update(self, model_parameters):
        for i, param in enumerate(model_parameters):
            self.models[i].load_state_dict(param, strict=False)

    def update_models(self, models):
        self.models = models

    def update_neighbors(self, neighbors):
        self.neighbors = neighbors

    def save_model(self, path, cur_round=-1):
        assert self.models is not None

        path = os.path.join(path, 'global')
        os.makedirs(path, exist_ok=True)
        neighbor_ids = list(self.neighbors.keys())
        for i, model in enumerate(self.models):
            ckpt = {'cur_round': cur_round, 'model': model.state_dict()}
            torch.save(ckpt, os.path.join(path, 'global_model_{}.pt'.format(neighbor_ids[i])))

    def load_model(self, path):
        if getattr(self, 'models', None):
            round = None
            global_dir = os.path.join(path, 'global')
            client_dir = os.path.join(path, 'client')
            neighbor_ids = sorted([int(re.search(r'model_(\d+).pt', x).groups()[0]) for x in os.listdir(global_dir)])
            assert len(neighbor_ids) == len(self.models)
            for i, model in enumerate(self.models):
                cur_global_path = os.path.join(global_dir, 'global_model_{}.pt'.format(neighbor_ids[i]))
                cur_client_path = os.path.join(client_dir, 'client_model_{}.pt'.format(neighbor_ids[i]))
                if os.path.exists(cur_global_path):
                    model_ckpt = model.state_dict()
                    logger.info('Loading model from \'{}\''.format(cur_global_path))
                    global_ckpt = torch.load(cur_global_path, map_location=self.device)
                    model_ckpt.update(global_ckpt['model'])
                    if os.path.exists(cur_client_path):
                        logger.info('Updating model from \'{}\''.format(cur_client_path))
                        client_ckpt = torch.load(cur_client_path, map_location=self.device)
                        model_ckpt.update(client_ckpt['model'])
                    self.models[i].load_state_dict(model_ckpt)
                    round = global_ckpt['cur_round']
                else:
                    raise ValueError("The file {} does NOT exist".format(cur_global_path))
            return round

    def _compute_client_groups(self, models):
        tasks = [None for _ in range(self.client_num)]
        if self.task == 'pretrain':
            task_groups = [random.randint(0, len(self.pretrain_tasks) - 1) for _ in range(self.num_agg_groups)]
            grads = torch.stack([model['model_grads'] for model in models])
            clustering = AgglomerativeClustering(n_clusters=self.num_agg_groups,
                                                 affinity='cosine',
                                                 linkage='average').fit(grads)
            self.client_id2group = clustering.labels_
            task_ids = [task_groups[self.client_id2group[i]] for i in range(self.client_num)]
            tasks = [self.pretrain_tasks[i] for i in task_ids]
        else:
            grads = torch.stack([model['model_grads'] for model in models])
            distances = cosine_distances(grads, grads)
            self.client_id2topk = np.argsort(distances, axis=-1)[:, :self.num_agg_topk].tolist()
        return tasks

    def _avg_params(self, models, client_adj_norm):
        model_params = [copy.deepcopy(model['model_para']) for model in models]
        for k in model_params[0]:
            if isinstance(model_params[0][k], torch.FloatTensor):
                cur_params = torch.stack([param[k] for param in model_params])
                avg_params = None
                if cur_params.ndim == 4:
                    avg_params = torch.einsum('ij,jklm->iklm', client_adj_norm, cur_params)
                elif cur_params.ndim == 3:
                    avg_params = torch.einsum('ij,jkl->ikl', client_adj_norm, cur_params)
                elif cur_params.ndim == 2:
                    avg_params = torch.einsum('ij,jk->ik', client_adj_norm, cur_params)
                for i in range(len(model_params)):
                    model_params[i][k] = avg_params[i]
        return model_params

    def _avg_mlm_head_params(self, models):
        avg_model = copy.deepcopy(models[0]['contrast_monitor'].mlm_head_params)
        for key in avg_model:
            for i in range(len(models)):
                local_model = models[i]['contrast_monitor'].mlm_head_params
                if i == 0:
                    avg_model[key] = local_model[key] / len(models)
                else:
                    avg_model[key] += local_model[key] / len(models)
        return avg_model

    def _para_weighted_avg(self, models, recover_fun=None):
        tasks = [None for _ in range(self.client_num)]
        if self.config.federate.method in ['local', 'global']:
            model_params = {'model_para': [model['model_para'] for model in models]}
            return model_params, tasks

        if self.task == 'pretrain':
            contrast_stat = models[0]['contrast_monitor'].stat
            for model in models:
                assert model['contrast_monitor'].stat == contrast_stat
            self.contrast_monitor.update_stat(contrast_stat)

            if contrast_stat == 1:
                enc_hidden = [model['contrast_monitor'].enc_hidden for model in models]
                enc_hidden = [x for x in enc_hidden if x is not None]
                if len(enc_hidden) > 0:
                    avg_enc_hidden = {}
                    for k in enc_hidden[0]:
                        avg_enc_hidden[k] = torch.stack([hid[k] for hid in enc_hidden]).mean(dim=0)

                    if len(avg_enc_hidden) > 0:
                        self.contrast_monitor.update_enc_hidden(avg_enc_hidden)
                        avg_mlm_head_params = self._avg_mlm_head_params(models)
                        self.encoder.load_state_dict(avg_mlm_head_params, strict=False)
                        with torch.no_grad():
                            batch_size = self.config.data.batch_size
                            mlm_inputs = torch.stack([
                                self.contrast_monitor.enc_hidden[k] for k in avg_enc_hidden]).to(self.config.device)
                            preds = torch.cat([
                                self.encoder.cls.predictions(mlm_inputs[i: i + batch_size]).detach().cpu().argmax(dim=-1)
                                for i in range(0, len(mlm_inputs), batch_size)])
                        synth_tokens = {k: v for k, v in zip(avg_enc_hidden, preds)}
                        self.contrast_monitor.update_synth_tokens(synth_tokens)

                model_params = {'contrast_monitor': self.contrast_monitor}

            elif contrast_stat == 2:
                dec_hidden = [model['contrast_monitor'].dec_hidden for model in models]
                dec_out = [model['contrast_monitor'].dec_out for model in models]
                dec_hidden = {k + 1: v for k, v in enumerate(dec_hidden)}
                dec_out = {k + 1: v for k, v in enumerate(dec_out)}
                group_ids = {k + 1: v for k, v in enumerate(self.client_id2group)}
                self.contrast_monitor.update_dec_hidden(dec_hidden)
                self.contrast_monitor.update_dec_out(dec_out)
                self.contrast_monitor.update_group_ids(group_ids)
                model_params = {'contrast_monitor': self.contrast_monitor}

            elif contrast_stat == 3:
                # generate param weight matrix
                client_adj = torch.zeros(self.client_num, self.client_num)
                for i in range(self.client_num):
                    for j in range(self.client_num):
                        if self.client_id2group[i] == self.client_id2group[j]:
                            client_adj[i][j] = models[j]['sample_size'] * self.inside_weight
                        else:
                            client_adj[i][j] = models[j]['sample_size'] * self.outside_weight
                client_adj_norm = client_adj / client_adj.sum(dim=-1, keepdim=True)

                # aggregate model params
                model_params = {'model_para': self._avg_params(models, client_adj_norm), 'contrast_monitor': self.contrast_monitor}
                # generate group task and self.client_id2group
                tasks = self._compute_client_groups(models)
                logger.info('client_id2group: {}'.format({k + 1: v for k, v in enumerate(self.client_id2group)}))

        else:
            # generate self.client_id2topk and param weight matrix
            tasks = self._compute_client_groups(models)
            logger.info('client_id2topk: {}'.format({k + 1: v for k, v in enumerate(self.client_id2topk)}))
            client_adj = torch.zeros(self.client_num, self.client_num)
            for i in range(self.client_num):
                for j in range(self.client_num):
                    if j in self.client_id2topk[i]:
                        client_adj[i][j] = models[j]['sample_size'] * self.inside_weight
                    else:
                        client_adj[i][j] = models[j]['sample_size'] * self.outside_weight
            client_adj_norm = client_adj / client_adj.sum(dim=-1, keepdim=True)

            # aggregate model params
            model_params = {'model_para': self._avg_params(models, client_adj_norm)}

        return model_params, tasks
