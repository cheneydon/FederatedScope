import collections
import copy
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.dataloader_builder import WrapDataset
from federatedscope.core.auxiliaries.ReIterator import ReIterator
from federatedscope.core.monitors.metric_calculator import MetricCalculator
from federatedscope.contrib.trainers.context import MyContext
from federatedscope.contrib.auxiliaries.utils import AverageMeter
from federatedscope.contrib.data.squad import SquadResult

logger = logging.getLogger(__name__)


# Build your trainer here.
class MyTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False):
        self.cfg = config
        self.metric_calculator = MetricCalculator(config.eval.metrics)

        self.ctx = MyContext(model,
                             self.cfg,
                             data,
                             device,
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

    def store_vars(self, ctx):
        store_dict = dict()
        store_dict['data_batch'] = ctx.data_batch
        store_dict['batch_size'] = ctx.batch_size
        store_dict['loss_task'] = ctx.loss_task
        store_dict['loss_batch'] = ctx.loss_batch
        store_dict['loss_regular'] = ctx.loss_regular
        store_dict['y_true'] = ctx.y_true
        store_dict['y_prob'] = ctx.y_prob
        return store_dict

    def restore_vars(self, ctx, store_dict):
        for k, v in store_dict.items():
            setattr(ctx, k, v)

    def _hook_on_fit_start_init(self, ctx):
        # prepare model
        ctx.model.to(ctx.device)

        # prepare statistics
        setattr(ctx, "loss_batch_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "loss_regular_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "num_samples_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "{}_y_true".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_y_prob".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_squad_results".format(ctx.cur_data_split), [])
        setattr(ctx, "best_sel_val_metric", float('-inf'))
        setattr(ctx, "best_epoch", -1)
        setattr(ctx, "best_step", -1)
        setattr(ctx, "best_val_metrics", [])
        setattr(ctx, "best_test_metrics", [])

    def _hook_on_epoch_start(self, ctx):
        # prepare dataloader
        if ctx.get("{}_loader".format(ctx.cur_data_split)) is None:
            loader = get_dataloader(WrapDataset(ctx.get("{}_data".format(ctx.cur_data_split))), self.cfg)
            setattr(ctx, "{}_loader".format(ctx.cur_data_split), ReIterator(loader))
        elif not isinstance(ctx.get("{}_loader".format(ctx.cur_data_split)), ReIterator):
            setattr(ctx, "{}_loader".format(ctx.cur_data_split),
                    ReIterator(ctx.get("{}_loader".format(ctx.cur_data_split))))
        else:
            ctx.get("{}_loader".format(ctx.cur_data_split)).reset()

        # prepare loss aggregator
        ctx.loss_agg = AverageMeter()

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
            ctx.y_true = labels
            ctx.y_prob = outputs.logits

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
            ctx.y_true = None
            ctx.y_prob = None

        ctx.batch_size = len(token_ids)
        ctx.loss_batch = outputs.loss
        ctx.loss_agg.update(ctx.loss_batch, ctx.batch_size)

    def _hook_on_batch_backward(self, ctx):
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

        if (ctx.cur_batch_i + 1) % ctx.cfg.trainer.val_freq == 0 or ctx.cur_batch_i + 1 == ctx.num_train_batch:
            task = ctx.cfg.data.type
            sel_metric = ctx.cfg.eval.sel_metric
            store_dict = self.store_vars(ctx)
            val_metrics = self.evaluate('val')

            val_value = val_metrics.get('val_{}'.format(task))
            if val_value is None:
                val_value = val_metrics['val_{}'.format(sel_metric)]
            else:
                val_value = val_value[sel_metric]

            if val_value >= ctx.best_sel_val_metric:
                test_metrics = self.evaluate('test')
                self.restore_vars(ctx, store_dict)
                ctx.best_sel_val_metric = val_value
                ctx.best_epoch = ctx.cur_epoch_i
                ctx.best_step = ctx.cur_batch_i
                ctx.best_val_metrics = val_metrics
                ctx.best_test_metrics = test_metrics
                logger.info('Best sel val metric {} found'.format(ctx.best_sel_val_metric))

    def _hook_on_batch_end(self, ctx):
        # update statistics
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
