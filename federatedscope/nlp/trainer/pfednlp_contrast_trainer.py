import collections
import copy
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from federatedscope.register import register_trainer
from federatedscope.nlp.monitors.metric_calculator import MetricCalculator
from federatedscope.nlp.trainer.pfednlp_trainer import PFedNLPTrainer
from federatedscope.nlp.trainer.context import FedNLPContext
from federatedscope.nlp.trainer.utils import ContrastiveMonitor, AverageMeter
from federatedscope.nlp.dataset.data.squad import SquadResult
from federatedscope.nlp.dataset.data.newsqa import NewsQAResult

logger = logging.getLogger(__name__)


# Build your trainer here.
class PFedNLPContrastTrainer(PFedNLPTrainer):
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
        self.ctx.contrast_monitor = ContrastiveMonitor()

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

    def update_contrast_monitor(self, contrast_monitor):
        self.ctx.contrast_monitor = contrast_monitor

    def update_stat(self, ID):
        super().update_stat(ID)
        self.ctx.model.update_client_id(ID)

    @property
    def _in_contrast_prepare(self):
        return self.task != 'pretrain' and self.ctx.cur_data_split == 'train' and \
               self.ctx.contrast_monitor.stat == 1

    def _run_routine(self, mode, hooks_set, dataset_name=None):
        if dataset_name is None:
            dataset_name = mode
        self.ctx.append_mode(mode)
        self.ctx.track_used_dataset(dataset_name)

        raw_num_train_epoch, raw_num_train_batch = None, None
        if self._in_contrast_prepare:
            raw_num_train_epoch, raw_num_train_batch = self.ctx.num_train_epoch, self.ctx.num_train_batch
            batch_size = self.ctx.cfg.data.batch_size
            num_contrast_data = len(self.ctx.contrast_monitor.synth_tokens)
            self.ctx.num_train_epoch = 1
            self.ctx.num_train_batch = num_contrast_data // batch_size + bool(num_contrast_data % batch_size)
            self.ctx.num_train_batch_last_epoch = self.ctx.num_train_batch
            self.ctx.num_total_train_batch = self.ctx.num_train_epoch * self.ctx.num_train_batch

        for hook in hooks_set['on_fit_start']:
            hook(self.ctx)

        for epoch_i in range(self.ctx.get('num_{}_epoch'.format(dataset_name))):
            self.ctx.cur_epoch_i = epoch_i
            for hook in hooks_set['on_epoch_start']:
                hook(self.ctx)

            for batch_i in tqdm(range(self.ctx.get('num_{}_batch'.format(dataset_name))),
                                disable=not (self._in_contrast_prepare or self.ctx.cur_data_split == 'test')):
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

        if raw_num_train_epoch is not None and raw_num_train_batch is not None:
            self.ctx.num_train_epoch = raw_num_train_epoch
            self.ctx.num_train_batch = raw_num_train_batch
            self.ctx.num_train_batch_last_epoch = self.ctx.num_train_batch
            self.ctx.num_total_train_batch = self.ctx.num_train_epoch * self.ctx.num_train_batch

        self.ctx.pop_mode()
        self.ctx.reset_used_dataset()
        self.ctx.model.to(torch.device('cpu'))

    def train(self, target_data_split_name='train', hooks_set=None):
        hooks_set = self.hooks_in_train if hooks_set is None else hooks_set
        if self.ctx.get(
                f'{target_data_split_name}_data') is None and self.ctx.get(
                    f'{target_data_split_name}_loader') is None:
            raise ValueError(
                f'No {target_data_split_name}_data or {target_data_split_name}_loader in the trainer'
            )
        self._run_routine('train', hooks_set, target_data_split_name)

        return self.ctx.num_samples_train, self.get_model_para(), self.get_model_grads(), self.ctx.contrast_monitor, self.ctx.eval_metrics

    def parse_data(self, data):
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ['train_raw', 'train_contrast', 'val', 'test']:
                init_dict['{}_data'.format(mode)] = None
                init_dict['{}_loader'.format(mode)] = None
                init_dict['num_{}_data'.format(mode)] = 0
                init_dict['{}_encoded'.format(mode)] = None
                init_dict['{}_examples'.format(mode)] = None
                if data.get(mode, None) is not None:
                    if isinstance(data.get(mode)['dataloader'], DataLoader):
                        init_dict['{}_loader'.format(mode)] = data.get(mode)['dataloader']
                        init_dict['num_{}_data'.format(mode)] = len(data.get(mode)['dataloader'].dataset)
                        init_dict['{}_encoded'.format(mode)] = data.get(mode)['encoded']
                        init_dict['{}_examples'.format(mode)] = data.get(mode)['examples']

                        if mode == 'train_raw':
                            init_dict['train_data'] = None
                            init_dict['train_loader'] = data.get(mode)['dataloader']
                            init_dict['num_train_data'] = len(data.get(mode)['dataloader'].dataset)
                            init_dict['train_encoded'] = data.get(mode)['encoded']
                            init_dict['train_examples'] = data.get(mode)['examples']
                    else:
                        raise TypeError('Type {} is not supported.'.format(
                            type(data.get(mode))))
        else:
            raise TypeError('Type of data should be dict.')
        return init_dict

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        if self._in_contrast_prepare:
            setattr(ctx, 'train_loader', ctx.get('train_contrast_loader'))
        else:
            setattr(ctx, 'regular_loss_agg_{}'.format(ctx.cur_data_split), AverageMeter())
            setattr(ctx, 'contrast_loss_agg_{}'.format(ctx.cur_data_split), AverageMeter())
            setattr(ctx, 'train_loader', ctx.get('train_raw_loader'))

    def _hook_on_batch_forward(self, ctx):
        ctx.contrast_loss_batch = None
        if self.task == 'pretrain':
            token_ids = ctx.data_batch[self.pretrain_task]['token_ids']
            attention_mask = ctx.data_batch[self.pretrain_task]['attention_mask']
            labels = ctx.data_batch[self.pretrain_task]['labels']
            example_indices = ctx.data_batch[self.pretrain_task]['example_indices']

            outputs = ctx.model(
                input_ids=token_ids.to(ctx.device),
                attention_mask=attention_mask.to(ctx.device),
                labels=labels.to(ctx.device),
                pretrain_task=self.pretrain_task,
                example_indices=example_indices,
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
                    contrast_monitor=ctx.contrast_monitor,
                    in_contrast_prepare=self._in_contrast_prepare,
                    example_indices=example_indices,
                )
                if not self._in_contrast_prepare:
                    ctx.batch_size = len(token_ids)
                    ctx.loss_batch = outputs.loss
                    ctx.regular_loss_batch = outputs.regular_loss
                    ctx.contrast_loss_batch = outputs.contrast_loss
                    ctx.y_true = labels
                    ctx.y_pred = outputs.logits.argmax(dim=-1)

            elif self.task in {'squad', 'newsqa'}:
                outputs = ctx.model(
                    input_ids=token_ids.to(ctx.device),
                    token_type_ids=token_type_ids.to(ctx.device),
                    attention_mask=attention_mask.to(ctx.device),
                    start_positions=start_positions.to(ctx.device),
                    end_positions=end_positions.to(ctx.device),
                    contrast_monitor=ctx.contrast_monitor,
                    in_contrast_prepare=self._in_contrast_prepare,
                    example_indices=example_indices,
                )
                if not self._in_contrast_prepare:
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
                    ctx.regular_loss_batch = outputs.regular_loss
                    ctx.contrast_loss_batch = outputs.contrast_loss
                    ctx.y_true = torch.cat([start_positions, end_positions])
                    ctx.y_pred = torch.cat([out.argmax(dim=-1) for out in outputs.logits])

            elif self.task in {'cnndm', 'msqg'}:
                if ctx.cur_data_split != 'test':
                    outputs = ctx.model(
                        input_ids=token_ids.to(ctx.device),
                        token_type_ids=token_type_ids.to(ctx.device),
                        attention_mask=attention_mask.to(ctx.device),
                        labels=labels.to(ctx.device),
                        contrast_monitor=ctx.contrast_monitor,
                        in_contrast_prepare=self._in_contrast_prepare,
                        example_indices=example_indices,
                    )
                    if not self._in_contrast_prepare:
                        ctx.batch_size = len(labels)
                        ctx.loss_batch = outputs.loss
                        ctx.regular_loss_batch = outputs.regular_loss
                        ctx.contrast_loss_batch = outputs.contrast_loss
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

        if self._in_contrast_prepare:
            dec_out, dec_hidden, example_indices = outputs.logits, outputs.hidden_states, outputs.example_indices
            for ex, out in zip(example_indices, dec_out.detach().cpu()):
                ctx.contrast_monitor.update_dec_out(out, k=ex.item())
            for ex, hids in zip(example_indices, dec_hidden.detach().cpu()):
                ctx.contrast_monitor.update_dec_hidden(hids, k=ex.item())
        else:
            ctx.get('loss_agg_{}'.format(ctx.cur_data_split)).update(ctx.loss_batch.detach().item(), ctx.batch_size)
            if ctx.get('regular_loss_batch', None) is not None:
                ctx.get('regular_loss_agg_{}'.format(ctx.cur_data_split)).update(
                    ctx.regular_loss_batch.detach().item(), ctx.batch_size)
            if ctx.get('contrast_loss_batch', None) is not None:
                ctx.get('contrast_loss_agg_{}'.format(ctx.cur_data_split)).update(
                    ctx.contrast_loss_batch.detach().item(), ctx.batch_size)

    def _hook_on_batch_forward_regularizer(self, ctx):
        if self._in_contrast_prepare:
            return
        super()._hook_on_batch_forward_regularizer(ctx)

    def _hook_on_batch_backward(self, ctx):
        if self._in_contrast_prepare:
            return
        super()._hook_on_batch_backward(ctx)

    def _hook_on_batch_end(self, ctx):
        if self._in_contrast_prepare:
            return
        super()._hook_on_batch_end(ctx)
        ctx.regular_loss_batch = None
        ctx.contrast_loss_batch = None

    def _hook_on_fit_end(self, ctx):
        if self.task != 'pretrain' and ctx.cur_data_split == 'train':
            ctx.contrast_monitor.update_stat(ctx.contrast_monitor.stat + 1)
            return
        super()._hook_on_fit_end(ctx)

    def _store_ctx(self, ctx):
        store_dict = super()._store_ctx(ctx)
        store_dict['contrast_loss_batch'] = ctx.loss_batch
        return store_dict


def call_pfednlp_contrast_trainer(trainer_type):
    if trainer_type == 'pfednlp_contrast_trainer':
        trainer_builder = PFedNLPContrastTrainer
        return trainer_builder


register_trainer('pfednlp_contrast_trainer', call_pfednlp_contrast_trainer)
