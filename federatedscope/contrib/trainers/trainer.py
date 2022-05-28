import os
import os.path as osp
import collections
import copy
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.monitors.metric_calculator import MetricCalculator
from federatedscope.contrib.trainers.context import MyContext
from federatedscope.contrib.auxiliaries.utils import AverageMeter
from federatedscope.contrib.data.squad import SquadResult
from federatedscope.contrib.auxiliaries.utils import Statistics

logger = logging.getLogger(__name__)


# Build your trainer here.
class MyTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, tokenizer, device, config, only_for_eval=False):
        self.cfg = config
        self.metric_calculator = MetricCalculator(config.eval.metrics)

        self.ctx = MyContext(model=model,
                             cfg=self.cfg,
                             data=data,
                             tokenizer=tokenizer,
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

    def parse_data(self, data):
        """Populate "{}_data", "{}_loader", "num_{}_data", "{}_encoded", "{}_examples" for different modes
        """
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_data".format(mode)] = None
                init_dict["{}_loader".format(mode)] = None
                init_dict["num_{}_data".format(mode)] = 0
                init_dict["{}_encoded".format(mode)] = None
                init_dict["{}_examples".format(mode)] = None
                if data.get(mode, None) is not None:
                    if isinstance(data.get(mode)['dataloader'], DataLoader):
                        init_dict["{}_loader".format(mode)] = data.get(mode)['dataloader']
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode)['dataloader'].dataset)
                        init_dict["{}_encoded".format(mode)] = data.get(mode)['encoded']
                        init_dict["{}_examples".format(mode)] = data.get(mode)['examples']
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(mode))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def store_ctx(self, ctx):
        store_dict = dict()
        store_dict['cur_epoch_i'] = ctx.cur_epoch_i
        store_dict['cur_batch_i'] = ctx.cur_batch_i
        store_dict['best_sel_val_metric'] = ctx.best_sel_val_metric
        store_dict['best_epoch'] = ctx.best_epoch
        store_dict['best_step'] = ctx.best_step
        store_dict['best_val_metrics'] = ctx.best_val_metrics
        store_dict['best_test_metrics'] = ctx.best_test_metrics
        store_dict['loss_agg'] = ctx.loss_agg
        store_dict['accum_steps'] = ctx.accum_steps
        store_dict['true_batches'] = ctx.true_batches
        store_dict['normalization'] = ctx.normalization

        store_dict['data_batch'] = ctx.data_batch
        store_dict['batch_size'] = ctx.batch_size
        store_dict['loss_task'] = ctx.get('loss_task')
        store_dict['loss_batch'] = ctx.get('loss_batch')
        store_dict['loss_regular'] = ctx.get('loss_regular')
        store_dict['y_true'] = ctx.get('y_true')
        store_dict['y_prob'] = ctx.get('y_prob')
        return store_dict

    def restore_ctx(self, ctx, store_dict):
        for k, v in store_dict.items():
            setattr(ctx, k, v)

    def _hook_on_fit_start_init(self, ctx):
        # prepare model
        ctx.model.to(ctx.device)
        setattr(ctx, "loss_func", getattr(ctx, "{}_loss_func".format(ctx.cur_data_split), None))
        if ctx.loss_func is not None: ctx.loss_func.to(ctx.device)

        # prepare statistics
        setattr(ctx, "accum_steps", 0)
        setattr(ctx, "true_batches", [])
        setattr(ctx, "normalization", 0)
        setattr(ctx, "loss_agg", AverageMeter() if ctx.cfg.data.type != 'cnndm' else Statistics())

        setattr(ctx, "loss_batch_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "loss_regular_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "num_samples_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "{}_y_true".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_y_prob".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_squad_results".format(ctx.cur_data_split), [])
        setattr(ctx, "best_sel_val_metric", float('-inf') if ctx.cfg.eval.is_higher_better else float('inf'))
        setattr(ctx, "best_epoch", -1)
        setattr(ctx, "best_step", -1)
        setattr(ctx, "best_val_metrics", [])
        setattr(ctx, "best_test_metrics", [])

    def _hook_on_batch_forward(self, ctx):
        task = ctx.cfg.data.type
        if task == 'imdb':
            token_ids, token_type_ids, attention_mask, labels = [_.to(ctx.device) for _ in ctx.data_batch]
            outputs = ctx.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                config=ctx.cfg,
            )
            ctx.batch_size = len(token_ids)
            ctx.loss_batch = outputs.loss
            ctx.y_true = labels
            ctx.y_prob = outputs.logits
            ctx.loss_agg.update(ctx.loss_batch, ctx.batch_size)

        elif task == 'squad':
            token_ids, token_type_ids, attention_mask, start_positions, end_positions, example_indices = \
                [_.to(ctx.device) for _ in ctx.data_batch]
            outputs = ctx.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                config=ctx.cfg,
            )
            for i, example_idx in enumerate(example_indices):
                encoded_input = ctx.get('{}_encoded'.format(ctx.cur_data_split))[example_idx.item()]
                unique_id = int(encoded_input.unique_id)
                start_logits = outputs.logits[0][i].detach().cpu().tolist()
                end_logits = outputs.logits[1][i].detach().cpu().tolist()
                ctx.get('{}_squad_results'.format(ctx.cur_data_split)).append(
                        SquadResult(unique_id, start_logits, end_logits))
            ctx.batch_size = len(token_ids)
            ctx.loss_batch = outputs.loss
            ctx.y_true = None
            ctx.y_prob = None
            ctx.loss_agg.update(ctx.loss_batch, ctx.batch_size)

        elif task == 'cnndm':
            if ctx.cur_data_split == 'train':
                ctx.true_batches.append(ctx.data_batch)
                src, segs, mask_src, tgt = ctx.data_batch
                num_tokens = tgt[:, 1:].ne(ctx.symbols['PAD']).sum()
                ctx.normalization += num_tokens.item()
                ctx.accum_steps += 1
                if ctx.accum_steps == ctx.cfg.trainer.grad_accum_count:
                    self._gradient_accumulation(ctx)
                    ctx.true_batches = []
                    ctx.accum_steps = 0
                    ctx.normalization = 0
            else:
                src, segs, mask_src, tgt = [_.to(ctx.device) for _ in ctx.data_batch]
                outputs = ctx.model(
                    input_ids=src,
                    attention_mask=mask_src,
                    token_type_ids=segs,
                    target_ids=tgt,
                    config=ctx.cfg,
                )

                batch_stats = ctx.loss_func.monolithic_compute_loss(tgt, outputs.logits)
                batch_stats.n_docs = int(src.size(0))
                ctx.loss_agg.update(batch_stats)

            ctx.batch_size = len(tgt)
            ctx.loss_batch = torch.tensor(ctx.loss_agg.xent()) \
                if ctx.loss_agg.n_words > 0 else torch.tensor(0)
            ctx.y_true = None
            ctx.y_prob = None

    def _gradient_accumulation(self, ctx):
        grad_accum_count = ctx.cfg.trainer.grad_accum_count
        if grad_accum_count > 1:
            ctx.model.zero_grad()
        for batch in ctx.true_batches:
            src, segs, mask_src, tgt = [_.to(ctx.device) for _ in batch]
            if grad_accum_count == 1:
                ctx.model.zero_grad()
            outputs = ctx.model(
                input_ids=src,
                attention_mask=mask_src,
                token_type_ids=segs,
                target_ids=tgt,
                config=ctx.cfg,
            )

            batch_stats = ctx.loss_func.sharded_compute_loss(
                tgt, outputs.logits, ctx.cfg.trainer.generator_shard_size, ctx.normalization)
            batch_stats.n_docs = int(src.size(0))
            ctx.loss_agg.update(batch_stats)

            if grad_accum_count == 1:
                for o in ctx.optimizer:
                    o.step()

        if grad_accum_count > 1:
            for o in ctx.optimizer:
                o.step()

    def _validate_and_test(self, ctx):
        task = ctx.cfg.data.type
        sel_metric = ctx.cfg.eval.sel_metric
        store_dict = self.store_ctx(ctx)
        logger.info('Start evaluation')
        val_metrics = self.evaluate('val')
        self.restore_ctx(ctx, store_dict)
        ctx.val_metrics = val_metrics

        default_sel_metric = 'avg_loss'
        val_value = val_metrics['val_{}'.format(default_sel_metric)]
        if sel_metric is not None:
            if task == 'imdb':
                val_value = val_metrics['val_{}'.format(sel_metric)]
            elif task == 'squad':
                val_value = val_metrics.get('val_{}'.format(sel_metric))
                if val_value is None:
                    val_value = val_metrics['val_squad'][sel_metric]

        is_best = val_value > ctx.best_sel_val_metric if ctx.cfg.eval.is_higher_better else \
            val_value < ctx.best_sel_val_metric
        if is_best:
            store_dict = self.store_ctx(ctx)
            test_metrics = self.evaluate('test')
            self.restore_ctx(ctx, store_dict)

            ctx.best_sel_val_metric = val_value
            ctx.best_epoch = ctx.cur_epoch_i + 1
            ctx.best_step = (ctx.cur_batch_i + 1) // ctx.cfg.trainer.grad_accum_count
            ctx.best_val_metrics = val_metrics
            ctx.best_test_metrics = test_metrics
            metric_name = sel_metric if sel_metric is not None else default_sel_metric
            logger.info('Best selected val metric ({}) found: {}'.format(metric_name, ctx.best_sel_val_metric))
            logger.info('Test metrics: {}'.format(test_metrics))

    def _save(self, ctx):
        ckpt = {
            'model': ctx.model.state_dict(),
            'optim': ctx.optimizer,
            'epoch': ctx.cur_epoch_i + 1,
            'step': (ctx.cur_batch_i + 1) // ctx.cfg.trainer.grad_accum_count
        }
        ckpt_path = osp.join(ctx.cfg.trainer.save_dir, 'last_model.pt')
        logger.info('Saving checkpoint {}'.format(ckpt_path))
        torch.save(ckpt, ckpt_path)

    def _hook_on_batch_backward(self, ctx):
        cur_step = (ctx.cur_batch_i + 1) // ctx.cfg.trainer.grad_accum_count
        if ctx.cfg.data.type == 'cnndm':
            if ctx.accum_steps == 0:
                if cur_step % ctx.cfg.trainer.disp_freq == 0 or cur_step == ctx.cfg.trainer.train_steps:
                    logger.info('Step: [{}/{}]\t'
                                'LR (enc): {:.2e}\t'
                                'LR (dec): {:.2e}\t'
                                'Loss: {}\t'
                                .format(cur_step, ctx.cfg.trainer.train_steps, ctx.optimizer[0].learning_rate,
                                        ctx.optimizer[1].learning_rate, ctx.loss_agg.xent()))
        else:
            ctx.optimizer.zero_grad()
            ctx.loss_task.backward()
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
            ctx.optimizer.step()
            ctx.scheduler.step()

            if ctx.cur_batch_i == 0 or (ctx.cur_batch_i + 1) % ctx.cfg.trainer.disp_freq == 0 or \
                    ctx.cur_batch_i + 1 == ctx.num_train_batch:
                logger.info('Epoch: [{}/{}][{}/{}]\t'
                            'LR: {:.2e}\t'
                            'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                            .format(ctx.cur_epoch_i + 1, ctx.num_train_epoch, ctx.cur_batch_i + 1, ctx.num_train_batch,
                                    ctx.scheduler.get_last_lr()[0], loss=ctx.loss_agg))

        if cur_step > 0 and ctx.accum_steps == 0 and \
                cur_step % ctx.cfg.trainer.val_freq == 0 or ctx.cur_batch_i + 1 == ctx.num_train_batch:
            self._validate_and_test(ctx)
            # self._save(ctx)

    def _hook_on_batch_end(self, ctx):
        # update statistics
        if ctx.accum_steps == 0:
            setattr(
                ctx, "loss_batch_total_{}".format(ctx.cur_data_split),
                ctx.get("loss_batch_total_{}".format(ctx.cur_data_split)) +
                ctx.loss_batch.item() * ctx.batch_size)

            if ctx.get("loss_regular", None) is None or ctx.loss_regular == 0:
                loss_regular = 0.
            else:
                loss_regular = ctx.loss_regular.item()
            setattr(
                ctx, "loss_regular_total_{}".format(ctx.cur_data_split),
                ctx.get("loss_regular_total_{}".format(ctx.cur_data_split)) +
                loss_regular)
            setattr(
                ctx, "num_samples_{}".format(ctx.cur_data_split),
                ctx.get("num_samples_{}".format(ctx.cur_data_split)) +
                ctx.batch_size)

            # cache label for evaluate
            if ctx.y_true is not None:
                ctx.get("{}_y_true".format(ctx.cur_data_split)).append(
                    ctx.y_true.detach().cpu().numpy())
            if ctx.y_prob is not None:
                ctx.get("{}_y_prob".format(ctx.cur_data_split)).append(
                    ctx.y_prob.detach().cpu().numpy())

            # clean temp ctx
            ctx.data_batch = None
            ctx.batch_size = None
            ctx.loss_task = None
            ctx.loss_batch = None
            ctx.loss_regular = None
            ctx.y_true = None
            ctx.y_prob = None

    def _hook_on_fit_end(self, ctx):
        """Evaluate metrics.
        """
        if ctx.cur_data_split == 'train':
            logger.info('-' * 50)
            logger.info('Best sel val metric {} found in epoch {} step {}'
                        .format(ctx.best_sel_val_metric, ctx.best_epoch, ctx.best_step))
            logger.info('Best val metrics: {}'.format(ctx.best_val_metrics))
            logger.info('Best test metrics: {}'.format(ctx.best_test_metrics))
            logger.info('-' * 50)
        else:
            if len(ctx.get("{}_y_true".format(ctx.cur_data_split))) > 0:
                setattr(
                    ctx, "{}_y_true".format(ctx.cur_data_split),
                    np.concatenate(ctx.get("{}_y_true".format(ctx.cur_data_split))))
            if len(ctx.get("{}_y_prob".format(ctx.cur_data_split))) > 0:
                setattr(
                    ctx, "{}_y_prob".format(ctx.cur_data_split),
                    np.concatenate(ctx.get("{}_y_prob".format(ctx.cur_data_split))))
            results = self.metric_calculator.eval(ctx)
            setattr(ctx, 'eval_metrics', results)


def call_my_trainer(trainer_type):
    if trainer_type == 'mytrainer':
        trainer_builder = MyTrainer
        return trainer_builder
