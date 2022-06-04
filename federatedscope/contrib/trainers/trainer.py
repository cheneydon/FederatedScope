import os
import os.path as osp
import collections
import copy
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.monitors.metric_calculator import MetricCalculator, eval_acc
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

        self.state = 0
        self.best_sel_val_metric = float('inf')
        self.best_round = 0
        self.best_epoch = 0
        self.best_step = 0

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

    def _set_state(self, round_num):
        self.state = round_num

    def _init_test(self, ctx):
        if ctx.cur_data_split == 'train':
            models = []
            cur_task = ctx.cfg.data.type
            tasks = None
            if cur_task == 'imdb':
                tasks = ['imdb', 'squad', 'cnndm']
                num_samples = [1, 0, 0]
            elif cur_task == 'squad':
                tasks = ['squad', 'imdb', 'cnndm']
                num_samples = [1, 0, 0]
            elif cur_task == 'cnndm':
                tasks = ['cnndm', 'imdb', 'squad']
                num_samples = [1, 0, 0]

            for task in tasks:
                ckpt_path = 'exp/share_enc/sub_exp_20220602113915/ckpt/{}/last_model.pt'.format(task)
                logger.info('Loading ckpt from \'{}\''.format(ckpt_path))
                state_dict = torch.load(ckpt_path, map_location='cpu')['model']
                models.append(state_dict)

            training_set_size = sum(num_samples)

            def _check_key(remove_keys, cur_key):
                for k in remove_keys:
                    if k in cur_key:
                        return False
                return True

            avg_model = models[0]
            remove_keys = ctx.cfg.personalization.local_param
            num_preserve_keys = 0
            num_remove_keys = 0
            for key in avg_model:
                if not _check_key(remove_keys, key):
                    num_remove_keys += 1
                    continue
                num_preserve_keys += 1
                for i in range(len(models)):
                    local_sample_size = num_samples[i]
                    local_model = models[i]
                    weight = local_sample_size / training_set_size

                    if isinstance(local_model[key], torch.Tensor):
                        local_model[key] = local_model[key].float()
                    else:
                        local_model[key] = torch.FloatTensor(local_model[key])

                    if i == 0:
                        avg_model[key] = local_model[key] * weight
                    else:
                        avg_model[key] += local_model[key] * weight

            ctx.model.load_state_dict(avg_model)
            logger.info('==> Start test evaluation')
            test_metrics = self.evaluate('test')
            logger.info(test_metrics)

    def _store_ctx(self, ctx):
        store_dict = {}
        store_dict['data_batch'] = ctx.data_batch
        store_dict['batch_size'] = ctx.batch_size
        store_dict['loss_task'] = ctx.loss_task
        store_dict['loss_batch'] = ctx.loss_batch
        store_dict['loss_regular'] = ctx.loss_regular
        store_dict['y_true'] = ctx.y_true
        store_dict['y_prob'] = ctx.y_prob
        return store_dict

    def _restore_ctx(self, ctx, store_dict):
        for k, v in store_dict.items():
            setattr(ctx, k, v)

    def _hook_on_fit_start_init(self, ctx):
        # prepare model
        ctx.model.to(ctx.device)
        setattr(ctx, "loss_func", getattr(ctx, "{}_loss_func".format(ctx.cur_data_split), None))
        if ctx.loss_func is not None:
            ctx.loss_func.to(ctx.device)

        # prepare statistics
        setattr(ctx, "loss_agg_{}".format(ctx.cur_data_split), AverageMeter()
                if ctx.cfg.data.type != 'cnndm' else Statistics())
        setattr(ctx, "loss_agg_report_{}".format(ctx.cur_data_split), Statistics())
        setattr(ctx, "loss_batch_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "loss_regular_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "num_samples_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "{}_y_true".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_y_prob".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_squad_results".format(ctx.cur_data_split), [])
        setattr(ctx, 'accum_steps', 0)
        setattr(ctx, 'true_batches', [])
        setattr(ctx, 'normalization', 0)

        if ctx.cfg.trainer.test_only:
            self._init_test(ctx)

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
            ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).update(ctx.loss_batch, ctx.batch_size)

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
            ctx.y_true = torch.cat([start_positions, end_positions])
            ctx.y_prob = torch.cat(outputs.logits)
            ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).update(ctx.loss_batch, ctx.batch_size)

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
                ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).update(batch_stats)
                ctx.get('loss_agg_report_{}'.format(ctx.cur_data_split)).update(batch_stats)

            ctx.batch_size = len(tgt)
            ctx.loss_batch = torch.tensor(ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).xent()) \
                if ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).n_words > 0 else torch.tensor(0)
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
            ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).update(batch_stats)
            ctx.get('loss_agg_report_{}'.format(ctx.cur_data_split)).update(batch_stats)

            if grad_accum_count == 1:
                for o in ctx.optimizer:
                    o.step()

        if grad_accum_count > 1:
            for o in ctx.optimizer:
                o.step()

    def _validate_and_test(self, ctx):
        # self._save(ctx, 'last_model.pt')
        logger.info('==> Start val evaluation')
        store_ctx = self._store_ctx(ctx)
        val_metrics = self.evaluate('val')
        self._restore_ctx(ctx, store_ctx)

        sel_val_metric = val_metrics['val_avg_loss']
        if sel_val_metric < self.best_sel_val_metric:
            self.best_sel_val_metric = sel_val_metric
            self.best_round = self.state + 1
            self.best_epoch = ctx.cur_epoch_i + 1
            self.best_step = (ctx.cur_batch_i + 1) // ctx.cfg.trainer.grad_accum_count
            logger.info('Best val_avg_loss found: {}'.format(self.best_sel_val_metric))
            self._save(ctx, 'best_model.pt')

        if self.state + 1 == ctx.cfg.federate.total_round_num:
            logger.info('==> Start test evaluation')
            store_ctx = self._store_ctx(ctx)
            # test_metrics = self.evaluate('test')
            # logger.info('Test metrics (last): {}'.format(test_metrics))

            raw_state_dict = copy.deepcopy(ctx.model.state_dict())
            best_ckpt_path = osp.join(ctx.cfg.trainer.save_dir, 'best_model.pt')
            best_state_dict = torch.load(best_ckpt_path, map_location='cpu')['model']
            ctx.model.load_state_dict(best_state_dict)
            test_metrics = self.evaluate('test')
            ctx.model.load_state_dict(raw_state_dict)
            logger.info('Test metrics (best): {}'.format(test_metrics))
            self._restore_ctx(ctx, store_ctx)

    def _save(self, ctx, ckpt_name):
        if ctx.cfg.trainer.save_dir:
            ckpt = {
                'model': ctx.model.state_dict(),
                # 'optim': ctx.optimizer,
                'round': self.state + 1,
                'epoch': ctx.cur_epoch_i + 1,
                'step': (ctx.cur_batch_i + 1) // ctx.cfg.trainer.grad_accum_count
            }
            ckpt_path = osp.join(ctx.cfg.trainer.save_dir, ckpt_name)
            # logger.info('Saving checkpoint {}'.format(ckpt_path))
            torch.save(ckpt, ckpt_path)

    def _hook_on_batch_backward(self, ctx):
        grad_accum_count = ctx.cfg.trainer.grad_accum_count
        cur_step = (ctx.cur_batch_i + 1) // grad_accum_count
        cur_task = ctx.cfg.data.type
        if cur_task != 'cnndm':
            ctx.accum_steps += 1
            ctx.loss_task = ctx.loss_task / grad_accum_count
            ctx.loss_task.backward()
            if ctx.accum_steps == grad_accum_count:
                if ctx.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
                ctx.optimizer.step()
                ctx.scheduler.step()
                ctx.optimizer.zero_grad()
                ctx.accum_steps = 0

        if cur_step > 0 and ctx.accum_steps == 0:
            if cur_step > 1:
                if cur_step % ctx.cfg.trainer.disp_freq == 0 or \
                        ctx.cur_batch_i + 1 == ctx.num_train_batch:
                    if cur_task == 'cnndm':
                        logger.info('Epoch: [{}/{}][{}/{}]\t'
                                    'LR (enc): {:.2e}\t'
                                    'LR (dec): {:.2e}\t'
                                    'Acc: {:.4f} ({:.4f})\t'
                                    'Loss: {:.4f} ({:.4f})\t'
                                    .format(ctx.cur_epoch_i + 1,
                                            ctx.num_train_epoch,
                                            cur_step,
                                            ctx.cfg.trainer.train_steps,
                                            ctx.optimizer[0].learning_rate,
                                            ctx.optimizer[1].learning_rate,
                                            ctx.get('loss_agg_report_{}'.format(ctx.cur_data_split)).accuracy(),
                                            ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).accuracy(),
                                            ctx.get('loss_agg_report_{}'.format(ctx.cur_data_split)).xent(),
                                            ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).xent()))
                    else:
                        y_true = ctx.y_true.detach().cpu().numpy()[:, None]
                        y_pred = np.argmax(ctx.y_prob.detach().cpu().numpy(), axis=-1)[:, None]
                        total_y_true = np.concatenate(ctx.get('{}_y_true'.format(ctx.cur_data_split)))[:, None]
                        total_y_pred = np.argmax(np.concatenate(
                            ctx.get('{}_y_prob'.format(ctx.cur_data_split))), axis=-1)[:, None]

                        logger.info('Epoch: [{}/{}][{}/{}]\t'
                                    'LR: {:.2e}\t'
                                    'Acc: {:.4f} ({:.4f})\t'
                                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                                    .format(ctx.cur_epoch_i + 1,
                                            ctx.num_train_epoch,
                                            cur_step,
                                            ctx.cfg.trainer.train_steps,
                                            ctx.scheduler.get_last_lr()[0],
                                            eval_acc(y_true, y_pred),
                                            eval_acc(total_y_true, total_y_pred),
                                            loss=ctx.get('loss_agg_{}'.format(ctx.cur_data_split))))

            if cur_task == 'cnndm':
                setattr(ctx, 'loss_agg_report_{}'.format(ctx.cur_data_split), Statistics())

            # if ctx.cur_batch_i + 1 == ctx.num_train_batch:
            #     if (self.state + 1) % ctx.cfg.eval.val_freq == 0 or \
            #             self.state + 1 == ctx.cfg.federate.total_round_num:
            #         self._validate_and_test(ctx)

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
            if self.state + 1 == ctx.cfg.federate.total_round_num:
                logger.info('Best val_avg_loss metric {} found in: round {} epoch {} step {}'
                            .format(self.best_sel_val_metric, self.best_round, self.best_epoch, self.best_step))
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
