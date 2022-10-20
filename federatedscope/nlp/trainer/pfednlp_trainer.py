import os
import os.path as osp
import collections
import copy
import logging
import re
import torch
import numpy as np
from collections import OrderedDict
from federatedscope.register import register_trainer
from federatedscope.core.auxiliaries.utils import filter_by_specified_keywords
from federatedscope.nlp.monitors.metric_calculator import MetricCalculator, eval_acc
from federatedscope.nlp.trainer.fednlp_trainer import FedNLPTrainer
from federatedscope.nlp.trainer.context import FedNLPContext
from federatedscope.nlp.dataset.data.squad import SquadResult
from federatedscope.nlp.dataset.data.newsqa import NewsQAResult

logger = logging.getLogger(__name__)


# Build your trainer here.
class PFedNLPTrainer(FedNLPTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False):
        self.cfg = config
        self.metric_calculator = MetricCalculator(config.eval.metrics)
        self.task = config.model.task
        self.pretrain_task = None
        self.ID = None
        self.load_ckpt = True
        self.pred_file, self.src_file, self.tgt_file = None, None, None
        self.finish_eval = False

        self.ctx = FedNLPContext(model=model,
                                 cfg=self.cfg,
                                 data=data,
                                 device=device,
                                 init_dict=self.parse_data(data))
        self.ctx.init_params = copy.deepcopy(model.state_dict())

        # Atomic operation during training/evaluation
        self.hooks_in_train = collections.defaultdict(list)

        # By default, use the same trigger keys
        self.hooks_in_eval = copy.deepcopy(self.hooks_in_train)

        # register necessary hooks into self.hooks_in_train and self.hooks_in_eval
        if not only_for_eval:
            self.register_default_hooks_train()
        self.register_default_hooks_eval()

        if self.cfg.federate.mode == 'distributed':
            self.print_trainer_meta_info()
        else:
            # in standalone mode, by default, we print the trainer info only once for better logs readability
            pass

    def update(self, model_parameters):
        super().update(model_parameters)
        self.ctx.init_params = copy.deepcopy(self.ctx.model.state_dict())

    def update_task(self, task):
        self.pretrain_task = task

    def get_model_grads(self, filter_keywords=None):
        if filter_keywords is None:
            filter_keywords = self.ctx.cfg.personalization.local_param
        grads = {}
        for n, p2 in self.ctx.model.state_dict().items():
            if filter_by_specified_keywords(n, filter_keywords):  # preserve
                grads[n] = p2 - self.ctx.init_params[n]
        return grads

    def _load_model(self, ctx):
        load_path = ctx.cfg.federate.load_from
        global_dir = os.path.join(load_path, 'global')
        client_dir = os.path.join(load_path, 'client')
        global_ckpt_path = os.path.join(global_dir, 'global_model_{}.pt'.format(self.ID))
        client_ckpt_path = os.path.join(client_dir, 'client_model_{}.pt'.format(self.ID))
        if os.path.exists(global_ckpt_path):
            model_ckpt = ctx.model.state_dict()
            logger.info('Loading model from \'{}\''.format(global_ckpt_path))
            global_ckpt = torch.load(global_ckpt_path, map_location='cpu')['model']
            model_ckpt.update(global_ckpt)
            if os.path.exists(client_ckpt_path):
                logger.info('Updating model from \'{}\''.format(client_ckpt_path))
                client_ckpt = torch.load(client_ckpt_path, map_location='cpu')['model']
                model_ckpt.update(client_ckpt)
            ctx.model.load_state_dict(model_ckpt)
        else:
            raise RuntimeError('Checkpoint NOT found in \'{}\''.format(global_ckpt_path))

    def _save_model(self, ctx):
        if len(ctx.cfg.personalization.local_param) > 0:
            model_ckpt = OrderedDict({k: v for k, v in ctx.model.state_dict().items()
                                      if re.search('|'.join(ctx.cfg.personalization.local_param), k) is not None})
            ckpt = {
                'model': model_ckpt,
                'epoch': ctx.cur_epoch_i + 1,
                'batch': ctx.cur_batch_i + 1,
            }
            save_dir = osp.join(ctx.cfg.federate.save_to, 'client')
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = osp.join(save_dir, 'client_model_{}.pt'.format(self.ID))
            torch.save(ckpt, ckpt_path)

    def train(self, target_data_split_name='train', hooks_set=None):
        hooks_set = self.hooks_in_train if hooks_set is None else hooks_set
        if self.ctx.get(
                f'{target_data_split_name}_data') is None and self.ctx.get(
                    f'{target_data_split_name}_loader') is None:
            raise ValueError(
                f'No {target_data_split_name}_data or {target_data_split_name}_loader in the trainer'
            )
        self._run_routine('train', hooks_set, target_data_split_name)

        return self.ctx.num_samples_train, self.get_model_para(), self.get_model_grads(), self.ctx.eval_metrics

    def _hook_on_batch_forward(self, ctx):
        if self.task == 'pretrain':
            token_ids = ctx.data_batch[self.pretrain_task]['token_ids']
            attention_mask = ctx.data_batch[self.pretrain_task]['attention_mask']
            labels = ctx.data_batch[self.pretrain_task]['labels']

            outputs = ctx.model(
                input_ids=token_ids.to(ctx.device),
                attention_mask=attention_mask.to(ctx.device),
                labels=labels.to(ctx.device),
                pretrain_task=self.pretrain_task,
            )
            ctx.batch_size = len(token_ids)
            ctx.loss_batch = outputs.loss
            if self.pretrain_task == 'mlm':
                ctx.y_true = labels
            elif self.pretrain_task == 'denoise':
                ctx.y_true = labels[:, 1:]
            count_idx = ctx.y_true.ne(-100) & ctx.y_true.ne(ctx.padding_idx)
            ctx.y_true = ctx.y_true[count_idx]
            ctx.y_pred = outputs.logits.argmax(dim=-1)[count_idx]

        else:
            token_ids = ctx.data_batch.get('token_ids', None)
            token_type_ids = ctx.data_batch.get('token_type_ids', None)
            attention_mask = ctx.data_batch.get('attention_mask', None)
            labels = ctx.data_batch.get('labels', None)
            start_positions = ctx.data_batch.get('start_positions', None)
            end_positions = ctx.data_batch.get('end_positions', None)
            example_indices = ctx.data_batch.get('example_indices', None)

            if self.task in {'imdb', 'agnews'}:
                outputs = ctx.model(
                    input_ids=token_ids.to(ctx.device),
                    token_type_ids=token_type_ids.to(ctx.device),
                    attention_mask=attention_mask.to(ctx.device),
                    labels=labels.to(ctx.device),
                )
                ctx.batch_size = len(token_ids)
                ctx.loss_batch = outputs.loss
                ctx.y_true = labels
                ctx.y_pred = outputs.logits.argmax(dim=-1)

            elif self.task in {'squad', 'newsqa'}:
                outputs = ctx.model(
                    input_ids=token_ids.to(ctx.device),
                    token_type_ids=token_type_ids.to(ctx.device),
                    attention_mask=attention_mask.to(ctx.device),
                    start_positions=start_positions.to(ctx.device),
                    end_positions=end_positions.to(ctx.device),
                )
                for i, example_idx in enumerate(example_indices):
                    encoded_input = ctx.get('{}_encoded'.format(ctx.cur_data_split))[example_idx.item()]
                    unique_id = int(encoded_input.unique_id)
                    start_logits = outputs.logits[0][i].detach().cpu().tolist()
                    end_logits = outputs.logits[1][i].detach().cpu().tolist()
                    if ctx.cur_data_split != 'train':
                        if self.task == 'squad':
                            ctx.get('{}_squad_results'.format(ctx.cur_data_split)).append(
                                    SquadResult(unique_id, start_logits, end_logits))
                        elif self.task == 'newsqa':
                            ctx.get('{}_newsqa_results'.format(ctx.cur_data_split)).append(
                                    NewsQAResult(unique_id, start_logits, end_logits))

                ctx.batch_size = len(token_ids)
                ctx.loss_batch = outputs.loss
                ctx.y_true = torch.cat([start_positions, end_positions])
                ctx.y_pred = torch.cat([out.argmax(dim=-1) for out in outputs.logits])

            elif self.task in {'cnndm', 'msqg'}:
                if ctx.cur_data_split != 'test':
                    outputs = ctx.model(
                        input_ids=token_ids.to(ctx.device),
                        token_type_ids=token_type_ids.to(ctx.device),
                        attention_mask=attention_mask.to(ctx.device),
                        labels=labels.to(ctx.device),
                    )
                    ctx.batch_size = len(labels)
                    ctx.loss_batch = outputs.loss
                    ctx.y_pred = outputs.logits.argmax(dim=-1)
                    ctx.y_true = labels[:, 1:]
                    non_padding_idx = ctx.y_true.ne(ctx.padding_idx)
                    ctx.y_true = ctx.y_true[non_padding_idx]
                    ctx.y_pred = ctx.y_pred[non_padding_idx]

                else:
                    outputs = ctx.model.generate(
                        input_ids=token_ids.to(ctx.device),
                        token_type_ids=token_type_ids.to(ctx.device),
                        attention_mask=attention_mask.to(ctx.device),
                    )
                    # save to file
                    out_str = ctx.tokenizer.batch_decode(outputs)
                    src_str = ctx.tokenizer.batch_decode(token_ids)
                    ref_str = ctx.tokenizer.batch_decode(labels)
                    for out, src, ref in zip(out_str, src_str, ref_str):
                        out = self._remove_special_tokens(out)
                        src = self._remove_special_tokens(src)
                        ref = self._remove_special_tokens(ref)
                        self.pred_file.write(out + '\n')
                        self.src_file.write(src + '\n')
                        self.tgt_file.write(ref + '\n')
                    self.pred_file.flush()
                    self.src_file.flush()
                    self.tgt_file.flush()

                    ctx.batch_size = len(labels)
                    ctx.y_pred = outputs
                    ctx.y_true = labels[:, 1:]
                    return

        ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).update(ctx.loss_batch.detach().item(), ctx.batch_size)

    def _hook_on_batch_backward(self, ctx):
        cur_step = (ctx.cur_batch_i + 1) // ctx.grad_accum_count
        ctx.accum_steps += 1
        ctx.loss_task = ctx.loss_task / ctx.grad_accum_count
        ctx.loss_task.backward()

        if ctx.accum_steps == ctx.grad_accum_count:
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
            ctx.optimizer.step()
            ctx.scheduler.step()
            ctx.optimizer.zero_grad()
            ctx.accum_steps = 0

        if cur_step > 0 and ctx.accum_steps == 0:
            if cur_step > 1 and (cur_step % ctx.cfg.trainer.disp_freq == 0 or ctx.cur_batch_i + 1 == ctx.num_train_batch):
                y_true = ctx.y_true.detach().cpu().numpy()
                y_pred = ctx.y_pred.detach().cpu().numpy()
                if y_true.ndim == 1:
                    y_true = np.expand_dims(y_true, axis=-1)
                if y_pred.ndim == 1:
                    y_pred = np.expand_dims(y_pred, axis=-1)
                cur_acc = eval_acc(y_true, y_pred)

                log_str = 'Epoch: [{}/{}][{}/{}]\t' \
                          'LR: {:.2e}\t' \
                          'Acc: {:.4f}\t' \
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(ctx.cur_epoch_i + 1,
                                                                         ctx.num_train_epoch,
                                                                         cur_step,
                                                                         ctx.cfg.trainer.train_steps,
                                                                         ctx.scheduler.get_last_lr()[0],
                                                                         cur_acc,
                                                                         loss=ctx.get('loss_agg_{}'.format(ctx.cur_data_split)))

                if ctx.get('regular_loss_batch', None) is not None and ctx.get('contrast_loss_batch', None) is not None:
                    log_str += '\tRegular loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                        loss=ctx.get('regular_loss_agg_{}'.format(ctx.cur_data_split)))
                    log_str += '\tContrastive loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                        loss=ctx.get('contrast_loss_agg_{}'.format(ctx.cur_data_split)))
                if self.task == 'pretrain':
                    log_str += '\t({})'.format(self.pretrain_task)
                logger.info(log_str)

            if ctx.cur_batch_i + 1 == ctx.num_train_batch:
                if ctx.cfg.federate.save_to:
                    self._save_model(ctx)


def call_pfednlp_trainer(trainer_type):
    if trainer_type in {'pfednlp_trainer', 'spfl_trainer', 'percfl_trainer'}:
        trainer_builder = PFedNLPTrainer
        return trainer_builder


register_trainer('pfednlp_trainer', call_pfednlp_trainer)
register_trainer('spfl_trainer', call_pfednlp_trainer)
register_trainer('percfl_trainer', call_pfednlp_trainer)
