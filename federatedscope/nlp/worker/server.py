import os.path as osp
import logging
import copy
import numpy as np
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.core.worker.server import Server
from federatedscope.nlp.trainer.utils import ContrastiveMonitor

logger = logging.getLogger(__name__)


class FedNLPServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):

        super().__init__(ID=ID,
                         state=state,
                         config=config,
                         data=data,
                         model=model,
                         client_num=client_num,
                         total_round_num=total_round_num,
                         device=device,
                         strategy=strategy,
                         **kwargs)

    def check_and_move_on(self, check_eval_result=False):
        if check_eval_result:
            # all clients are participating in evaluation
            minimal_number = self.client_num
        else:
            # sampled clients are participating in training
            minimal_number = self.sample_client_num

        if self.check_buffer(self.state, minimal_number, check_eval_result):
            if not check_eval_result:  # in the training process
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                for model_idx in range(self.model_num):
                    model = self.models[model_idx]
                    aggregator = self.aggregators[model_idx]
                    msg_list = list()
                    for client_id in train_msg_buffer:
                        if self.model_num == 1:
                            msg_list.append(train_msg_buffer[client_id])
                        else:
                            train_data_size, model_para_multiple = train_msg_buffer[client_id]
                            msg_list.append((train_data_size, model_para_multiple[model_idx]))

                    # Aggregate
                    agg_info = {
                        'client_feedback': msg_list,
                        'recover_fun': self.recover_fun,
                    }
                    result = aggregator.aggregate(agg_info)
                    model.load_state_dict(result, strict=False)

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server #{:d}: Starting evaluation at the end of round {:d}.'
                        .format(self.ID, self.state + 1))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        '----------- Starting a new training round (Round #{:d}/{:d}) -------------'
                        .format(self.state + 1, self._cfg.federate.total_round_num))
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info(
                        'Server #{:d}: Training is finished! Starting evaluation.'
                        .format(self.ID))
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                self.check_and_save()

    def save_best_results(self):
        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(osp.join(self._cfg.federate.save_to, 'global_model.pt'), self.state)

    def merge_eval_results_from_all_clients(self, final_round=False):
        state = self.state if not final_round else self.state - 1
        eval_msg_buffer = self.msg_buffer['eval'][state]

        if 'group_avg' in self._cfg.eval.report:
            metrics_all_clients = eval_msg_buffer
        else:
            metrics_all_clients = dict()
            for each_client in eval_msg_buffer:
                client_eval_results = eval_msg_buffer[each_client]
                for key in client_eval_results.keys():
                    res = client_eval_results[key]
                    if isinstance(res, dict):
                        for k, v in res.items():
                            cur_key = key + '_' + k
                            if key not in metrics_all_clients:
                                metrics_all_clients[cur_key] = list()
                            metrics_all_clients[cur_key].append(float(v))
                    else:
                        if key not in metrics_all_clients:
                            metrics_all_clients[key] = list()
                        metrics_all_clients[key].append(float(res))
        formatted_logs = self._monitor.format_eval_res(
            metrics_all_clients,
            rnd=self.state + 1,
            role='Server #',
            forms=self._cfg.eval.report)
        logger.info(formatted_logs)
        self.save_formatted_results(formatted_logs)
        return formatted_logs

    def eval(self):
        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all internal models;
            # for other cases such as ensemble, override the eval function
            for i in range(self.model_num):
                trainer = self.trainers[i]
                # Preform evaluation in server
                metrics = {}
                for split in self._cfg.eval.split:
                    eval_metrics = trainer.evaluate(
                        target_data_split_name=split)
                    metrics.update(**eval_metrics)
                formatted_eval_res = self._monitor.format_eval_res(
                    metrics,
                    rnd=self.state + 1,
                    role='Server #',
                    forms=self._cfg.eval.report,
                    return_raw=self._cfg.federate.make_global_eval)
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                self.save_formatted_results(formatted_eval_res)
                logger.info(formatted_eval_res)
            self.check_and_save()
        else:
            # Preform evaluation in clients
            self.broadcast_model_para(msg_type='evaluate')


class PFedNLPServer(FedNLPServer):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):

        super().__init__(ID=ID,
                         state=state,
                         config=config,
                         data=data,
                         model=model,
                         client_num=client_num,
                         total_round_num=total_round_num,
                         device=device,
                         strategy=strategy,
                         **kwargs)

        self.models = [copy.deepcopy(self.model) for _ in range(self.client_num)]
        self.tasks = [config.data.pretrain_tasks[0] if config.data.pretrain_tasks else None for _ in range(self.client_num)]
        self.aggregator.update_models(self.models)
        self.aggregator.update_neighbors(self.comm_manager.neighbors)
        if self._cfg.federate.restore_from != '':
            cur_round = self.aggregator.load_model(self._cfg.federate.restore_from)
            logger.info("Restored the model from {}-th round's ckpt".format(cur_round))

    def check_and_move_on(self, check_eval_result=False):
        if check_eval_result:
            # all clients are participating in evaluation
            minimal_number = self.client_num
        else:
            # sampled clients are participating in training
            minimal_number = self.sample_client_num

        if self.check_buffer(self.state, minimal_number, check_eval_result):
            if not check_eval_result:  # in the training process
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                train_msg_buffer = dict(sorted(train_msg_buffer.items(), key=lambda x: x[0]))
                msg_list = list()
                for client_id in train_msg_buffer:
                    msg_list.append(train_msg_buffer[client_id])

                # Aggregate
                agg_info = {
                    'client_feedback': msg_list,
                    'recover_fun': self.recover_fun
                }
                avg_models, tasks = self.aggregator.aggregate(agg_info)
                self.tasks = tasks
                for i in range(self.client_num):
                    self.models[i].load_state_dict(avg_models[i], strict=False)

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server #{:d}: Starting evaluation at the end of round {:d}.'
                        .format(self.ID, self.state + 1))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        '----------- Starting a new training round (Round #{:d}/{:d}) -------------'
                        .format(self.state + 1, self._cfg.federate.total_round_num))
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info('Server #{:d}: Training is finished! Starting evaluation.'.format(self.ID))
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results, formatted_eval_res)
                self.check_and_save()

    def save_best_results(self):
        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(self._cfg.federate.save_to, self.state)

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1):
        if sample_client_num > 0:
            sample_ids = np.random.choice(np.arange(self.client_num), size=sample_client_num, replace=False).tolist()
        else:
            # broadcast to all clients
            sample_ids = list(range(self.client_num))

        receivers = list(self.comm_manager.neighbors.keys())
        model_para = [model.state_dict() for model in self.models]
        skip_broadcast = self._cfg.federate.method in ['local', 'global']
        if skip_broadcast:
            model_para = [{} for _ in self.models]

        for i in sample_ids:
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receivers[i],
                        state=self.state,
                        content={'model_para': model_para[i], 'task': self.tasks[i]}))


class PFedNLPContrastServer(FedNLPServer):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):

        super().__init__(ID=ID,
                         state=state,
                         config=config,
                         data=data,
                         model=model,
                         client_num=client_num,
                         total_round_num=total_round_num,
                         device=device,
                         strategy=strategy,
                         **kwargs)

        self.models = [copy.deepcopy(self.model) for _ in range(self.client_num)]
        self.tasks = [config.data.pretrain_tasks[0] if config.data.pretrain_tasks else None for _ in range(self.client_num)]
        self.aggregator.update_models(self.models)
        self.aggregator.update_neighbors(self.comm_manager.neighbors)
        self.contrast_monitor = ContrastiveMonitor()
        if self._cfg.federate.restore_from != '':
            cur_round = self.aggregator.load_model(self._cfg.federate.restore_from)
            logger.info("Restored the model from {}-th round's ckpt".format(cur_round))

    def check_and_move_on(self, check_eval_result=False):
        if check_eval_result:
            # all clients are participating in evaluation
            minimal_number = self.client_num
        else:
            # sampled clients are participating in training
            minimal_number = self.sample_client_num

        if self.check_buffer(self.state, minimal_number, check_eval_result):
            if not check_eval_result:  # in the training process
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                train_msg_buffer = dict(sorted(train_msg_buffer.items(), key=lambda x: x[0]))
                msg_list = list()
                for client_id in train_msg_buffer:
                    msg_list.append(train_msg_buffer[client_id])

                # Aggregate
                agg_info = {
                    'client_feedback': msg_list,
                    'recover_fun': self.recover_fun
                }
                avg_models, tasks = self.aggregator.aggregate(agg_info)
                self.tasks = tasks
                if 'contrast_monitor' in avg_models:
                    self.contrast_monitor = avg_models['contrast_monitor']
                    if self.contrast_monitor.stat == 3:
                        self.contrast_monitor.reset()
                if 'model_para' in avg_models:
                    for i in range(self.client_num):
                        self.models[i].load_state_dict(avg_models['model_para'][i], strict=False)

                if self._cfg.data.task == 'pretrain' and self.contrast_monitor.stat in {1, 2}:
                    self.msg_buffer['train'][self.state].clear()
                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                    return

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server #{:d}: Starting evaluation at the end of round {:d}.'
                        .format(self.ID, self.state + 1))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        '----------- Starting a new training round (Round #{:d}/{:d}) -------------'
                        .format(self.state + 1, self._cfg.federate.total_round_num))
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info('Server #{:d}: Training is finished! Starting evaluation.'.format(self.ID))
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results, formatted_eval_res)
                self.check_and_save()

    def save_best_results(self):
        if self._cfg.federate.save_to != '':
            self.aggregator.save_model(self._cfg.federate.save_to, self.state)

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1):
        if sample_client_num > 0:
            sample_ids = np.random.choice(np.arange(self.client_num), size=sample_client_num, replace=False).tolist()
        else:
            # broadcast to all clients
            sample_ids = list(range(self.client_num))

        receivers = list(self.comm_manager.neighbors.keys())
        model_para = [model.state_dict() for model in self.models]
        skip_broadcast = self._cfg.federate.method in ['local', 'global']
        if skip_broadcast:
            model_para = [{} for _ in self.models]

        for i in sample_ids:
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receivers[i],
                        state=self.state,
                        content={'model_para': model_para[i],
                                 'task': self.tasks[i],
                                 'contrast_monitor': self.contrast_monitor}))
