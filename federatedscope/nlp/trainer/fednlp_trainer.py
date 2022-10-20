import os
import os.path as osp
import collections
import copy
import logging
import re
import torch
import numpy as np
import codecs
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.nlp.monitors.metric_calculator import MetricCalculator, eval_acc
from federatedscope.nlp.trainer.utils import AverageMeter
from federatedscope.nlp.trainer.context import FedNLPContext
from federatedscope.nlp.dataset.data.squad import SquadResult
from federatedscope.nlp.dataset.data.newsqa import NewsQAResult

logger = logging.getLogger(__name__)


# Build your trainer here.
class FedNLPTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False):
        self.cfg = config
        self.metric_calculator = MetricCalculator(config.eval.metrics)
        self.task = config.model.task
        self.ID = None
        self.load_ckpt = True
        self.pred_file, self.src_file, self.tgt_file = None, None, None
        self.finish_eval = False

        self.ctx = FedNLPContext(model=model,
                                 cfg=self.cfg,
                                 data=data,
                                 device=device,
                                 init_dict=self.parse_data(data))

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

    def update_stat(self, ID):
        self.ID = ID
        if self.task in {'cnndm', 'msqg'}:
            pred_dir = osp.join(self.cfg.test.result_path, 'pred')
            src_dir = osp.join(self.cfg.test.result_path, 'src')
            tgt_dir = osp.join(self.cfg.test.result_path, 'tgt')
            self.ctx.pred_path = osp.join(pred_dir, '%d.txt' % ID)
            self.ctx.src_path = osp.join(src_dir, '%d.txt' % ID)
            self.ctx.tgt_path = osp.join(tgt_dir, '%d.txt' % ID)

            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(src_dir, exist_ok=True)
            os.makedirs(tgt_dir, exist_ok=True)
            self.pred_file = codecs.open(self.ctx.pred_path, 'w', 'utf-8')
            self.src_file = codecs.open(self.ctx.src_path, 'w', 'utf-8')
            self.tgt_file = codecs.open(self.ctx.tgt_path, 'w', 'utf-8')

    def parse_data(self, data):
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ['train', 'val', 'test']:
                init_dict['{}_data'.format(mode)] = None
                init_dict['{}_loader'.format(mode)] = None
                init_dict['num_{}_data'.format(mode)] = 0
                init_dict['{}_encoded'.format(mode)] = None
                init_dict['{}_examples'.format(mode)] = None
                if data.get(mode, None) is not None:
                    if isinstance(data.get(mode)['dataloader'], DataLoader):
                        init_dict['{}_loader'.format(mode)] = data.get(mode)['dataloader']
                        init_dict['num_{}_data'.format(mode)] = len(
                            data.get(mode)['dataloader'].dataset)
                        init_dict['{}_encoded'.format(mode)] = data.get(mode)['encoded']
                        init_dict['{}_examples'.format(mode)] = data.get(mode)['examples']
                    else:
                        raise TypeError('Type {} is not supported.'.format(
                            type(data.get(mode))))
        else:
            raise TypeError('Type of data should be dict.')
        return init_dict

    def _load_model(self, ctx):
        load_path = ctx.cfg.federate.load_from
        global_ckpt_path = osp.join(load_path, 'global_model.pt')
        client_ckpt_path = osp.join(load_path, 'client_model_{}.pt'.format(self.ID))
        if not osp.exists(global_ckpt_path):
            global_dir = os.path.join(load_path, 'global')
            client_dir = os.path.join(load_path, 'client')
            global_ckpt_path = os.path.join(global_dir, 'global_model_{}.pt'.format(self.ID))
            client_ckpt_path = os.path.join(client_dir, 'client_model_{}.pt'.format(self.ID))
            if not osp.exists(global_ckpt_path):
                raise RuntimeError('Checkpoint NOT found in \'{}\''.format(global_ckpt_path))

        model_ckpt = ctx.model.state_dict()
        logger.info('Loading model from \'{}\''.format(global_ckpt_path))
        global_ckpt = torch.load(global_ckpt_path, map_location='cpu')['model']
        model_ckpt.update(global_ckpt)
        if osp.exists(client_ckpt_path):
            logger.info('Updating model from \'{}\''.format(client_ckpt_path))
            client_ckpt = torch.load(client_ckpt_path, map_location='cpu')['model']
            model_ckpt.update(client_ckpt)
        ctx.model.load_state_dict({k: v for k, v in model_ckpt.items() if k in ctx.model.state_dict()})

    def _save_model(self, ctx):
        if len(ctx.cfg.personalization.local_param) > 0:
            model_ckpt = OrderedDict({k: v for k, v in ctx.model.state_dict().items()
                                      if re.search('|'.join(ctx.cfg.personalization.local_param), k) is not None})
            ckpt = {
                'model': model_ckpt,
                'epoch': ctx.cur_epoch_i + 1,
                'batch': ctx.cur_batch_i + 1,
            }
            save_dir = ctx.cfg.federate.save_to
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = osp.join(save_dir, 'client_model_{}.pt'.format(self.ID))
            torch.save(ckpt, ckpt_path)

    def _remove_special_tokens(self, sent):
        return sent.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', ''). \
            replace('[unused0]', '').replace('[unused3]', '').replace('[unused1]', ''). \
            replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()

    def _run_routine(self, mode, hooks_set, dataset_name=None):
        if self.finish_eval:
            return
        if dataset_name is None:
            dataset_name = mode
        self.ctx.append_mode(mode)
        self.ctx.track_used_dataset(dataset_name)

        for hook in hooks_set['on_fit_start']:
            hook(self.ctx)

        for epoch_i in range(self.ctx.get(
                'num_{}_epoch'.format(dataset_name))):
            self.ctx.cur_epoch_i = epoch_i
            for hook in hooks_set['on_epoch_start']:
                hook(self.ctx)

            for batch_i in tqdm(range(
                    self.ctx.get('num_{}_batch'.format(dataset_name))), disable=self.ctx.cur_data_split != 'test'):
                self.ctx.cur_batch_i = batch_i
                for hook in hooks_set['on_batch_start']:
                    hook(self.ctx)
                for hook in hooks_set['on_batch_forward']:
                    hook(self.ctx)
                if self.ctx.cur_mode == 'train':
                    for hook in hooks_set['on_batch_backward']:
                        hook(self.ctx)
                for hook in hooks_set['on_batch_end']:
                    hook(self.ctx)

                # Break in the final epoch
                if self.ctx.cur_mode == 'train' and epoch_i == self.ctx.num_train_epoch - 1:
                    if batch_i >= self.ctx.num_train_batch_last_epoch - 1:
                        break

            for hook in hooks_set['on_epoch_end']:
                hook(self.ctx)
        for hook in hooks_set['on_fit_end']:
            hook(self.ctx)

        self.ctx.pop_mode()
        self.ctx.reset_used_dataset()
        # Avoid memory leak
        if not self.cfg.federate.share_local_model:
            if torch is None:
                pass
            else:
                self.ctx.model.to(torch.device('cpu'))

    def _hook_on_fit_start_init(self, ctx):
        # prepare model
        ctx.model.to(ctx.device)
        if ctx.cur_data_split == 'train' and ctx.cfg.federate.load_from and self.load_ckpt:
            self._load_model(ctx)
            self.load_ckpt = False
        # prepare statistics
        setattr(ctx, 'loss_agg_{}'.format(ctx.cur_data_split), AverageMeter())
        setattr(ctx, 'loss_batch_total_{}'.format(ctx.cur_data_split), 0)
        setattr(ctx, 'loss_regular_total_{}'.format(ctx.cur_data_split), 0)
        setattr(ctx, 'num_samples_{}'.format(ctx.cur_data_split), 0)
        setattr(ctx, '{}_y_true'.format(ctx.cur_data_split), [])
        setattr(ctx, '{}_y_pred'.format(ctx.cur_data_split), [])
        setattr(ctx, '{}_squad_results'.format(ctx.cur_data_split), [])
        setattr(ctx, '{}_newsqa_results'.format(ctx.cur_data_split), [])
        setattr(ctx, 'accum_steps', 0)

    def _hook_on_batch_forward(self, ctx):
        token_ids = ctx.data_batch.get('token_ids', None)
        token_type_ids = ctx.data_batch.get('token_type_ids', None)
        attention_mask = ctx.data_batch.get('attention_mask', None)
        labels = ctx.data_batch.get('labels', None)
        start_positions = ctx.data_batch.get('start_positions', None)
        end_positions = ctx.data_batch.get('end_positions', None)
        example_indices = ctx.data_batch.get('example_indices', None)

        if self.task == 'pretrain':
            outputs = ctx.model(
                input_ids=token_ids.to(ctx.device),
                attention_mask=attention_mask.to(ctx.device),
                labels=labels.to(ctx.device),
            )
            ctx.batch_size = len(token_ids)
            ctx.loss_batch = outputs.loss
            pretrain_task = ctx.cfg.model.pretrain_task
            if pretrain_task == 'mlm':
                ctx.y_true = labels
            elif pretrain_task == 'denoise':
                ctx.y_true = labels[:, 1:]
            count_idx = ctx.y_true.ne(-100) & ctx.y_true.ne(ctx.padding_idx)
            ctx.y_true = ctx.y_true[count_idx]
            ctx.y_pred = outputs.logits.argmax(dim=-1)[count_idx]

        elif self.task in {'imdb', 'agnews'}:
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

                logger.info('Epoch: [{}/{}][{}/{}]\t'
                            'LR: {:.2e}\t'
                            'Acc: {:.4f}\t'
                            'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                            .format(ctx.cur_epoch_i + 1,
                                    ctx.num_train_epoch,
                                    cur_step,
                                    ctx.cfg.trainer.train_steps,
                                    ctx.scheduler.get_last_lr()[0],
                                    cur_acc,
                                    loss=ctx.get('loss_agg_{}'.format(ctx.cur_data_split))))

            if ctx.cur_batch_i + 1 == ctx.num_train_batch:
                if ctx.cfg.federate.method == 'local':
                    self._test(ctx)
                if ctx.cfg.federate.save_to:
                    self._save_model(ctx)

    def _hook_on_batch_end(self, ctx):
        # update statistics
        if ctx.get('loss_batch', None) is not None:
            setattr(
                ctx, 'loss_batch_total_{}'.format(ctx.cur_data_split),
                ctx.get('loss_batch_total_{}'.format(ctx.cur_data_split)) +
                ctx.loss_batch.item() * ctx.batch_size)

        if ctx.get('loss_regular', None) is None or ctx.loss_regular == 0:
            loss_regular = 0.
        else:
            loss_regular = ctx.loss_regular.item()
        setattr(
            ctx, 'loss_regular_total_{}'.format(ctx.cur_data_split),
            ctx.get('loss_regular_total_{}'.format(ctx.cur_data_split)) +
            loss_regular)
        setattr(
            ctx, 'num_samples_{}'.format(ctx.cur_data_split),
            ctx.get('num_samples_{}'.format(ctx.cur_data_split)) +
            ctx.batch_size)

        # cache label for evaluate
        if self.task in {'pretrain', 'squad', 'newsqa', 'cnndm', 'msqg'}:
            setattr(ctx, '{}_y_true'.format(ctx.cur_data_split), [ctx.y_true.detach().cpu().numpy()])
            setattr(ctx, '{}_y_pred'.format(ctx.cur_data_split), [ctx.y_pred.detach().cpu().numpy()])
        else:
            ctx.get('{}_y_true'.format(ctx.cur_data_split)).append(
                ctx.y_true.detach().cpu().numpy())
            ctx.get('{}_y_pred'.format(ctx.cur_data_split)).append(
                ctx.y_pred.detach().cpu().numpy())

        # clean temp ctx
        ctx.data_batch = None
        ctx.batch_size = None
        ctx.loss_task = None
        ctx.loss_batch = None
        ctx.loss_regular = None
        ctx.y_true = None
        ctx.y_pred = None

    def _hook_on_fit_end(self, ctx):
        if ctx.cur_data_split != 'train':
            setattr(ctx, '{}_y_true'.format(ctx.cur_data_split),
                    np.concatenate(ctx.get('{}_y_true'.format(ctx.cur_data_split))))
            setattr(ctx, '{}_y_pred'.format(ctx.cur_data_split),
                    np.concatenate(ctx.get('{}_y_pred'.format(ctx.cur_data_split))))
            results = self.metric_calculator.eval(ctx)
            setattr(ctx, 'eval_metrics', results)

        if ctx.cur_data_split == 'test' and not self.finish_eval:
            if self.pred_file is not None:
                self.pred_file.close()
            if self.src_file is not None:
                self.src_file.close()
            if self.tgt_file is not None:
                self.tgt_file.close()
            self.finish_eval = True

    def _store_ctx(self, ctx):
        store_dict = {}
        store_dict['data_batch'] = ctx.data_batch
        store_dict['batch_size'] = ctx.batch_size
        store_dict['loss_task'] = ctx.loss_task
        store_dict['loss_batch'] = ctx.loss_batch
        store_dict['loss_regular'] = ctx.loss_regular
        store_dict['y_true'] = ctx.y_true
        store_dict['y_pred'] = ctx.y_pred
        return store_dict

    def _restore_ctx(self, ctx, store_dict):
        for k, v in store_dict.items():
            setattr(ctx, k, v)

    def _test(self, ctx):
        logger.info('==> Start test evaluation')
        store_ctx = self._store_ctx(ctx)
        test_metrics = self.evaluate('test')
        logger.info('Test metrics before aggregation: {}'.format(test_metrics))
        self._restore_ctx(ctx, store_ctx)


def call_fednlp_trainer(trainer_type):
    if trainer_type == 'fednlp_trainer':
        trainer_builder = FedNLPTrainer
        return trainer_builder


register_trainer('fednlp_trainer', call_fednlp_trainer)
