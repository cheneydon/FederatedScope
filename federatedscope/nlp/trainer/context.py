import math
import logging
from federatedscope.core.trainers.context import Context
from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.model_builder import get_trainable_para_names
from federatedscope.core.auxiliaries.regularizer_builder import get_regularizer
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.nlp.trainer.utils import setup_tokenizer

logger = logging.getLogger(__name__)


class FedNLPContext(Context):
    def setup_vars(self):
        if self.cfg.backend == 'torch':
            self.trainable_para_names = get_trainable_para_names(self.model)
            self.criterion = get_criterion(self.cfg.criterion.type, self.device)
            self.regularizer = get_regularizer(self.cfg.regularizer.type)
            self.tokenizer = setup_tokenizer(self.cfg.model.bert_type)
            self.eval_metrics = None

            task = self.cfg.data.task
            if self.cfg.trainer.train_steps is not None:
                num_steps = self.cfg.trainer.train_steps * self.cfg.federate.total_round_num
            else:
                num_steps = len(self.train_loader) * self.cfg.federate.local_update_steps * \
                            self.cfg.federate.total_round_num

            if task in {'imdb', 'agnews', 'squad', 'newsqa'}:
                self.optimizer = get_optimizer(
                    self.cfg.optimizer.type,
                    self.model,
                    self.cfg.optimizer.lr,
                    weight_decay=self.cfg.optimizer.weight_decay,
                )
                self.scheduler = get_scheduler(
                    self.cfg.scheduler.type,
                    self.optimizer,
                    total_steps=num_steps,
                    warmup_steps=int(self.cfg.scheduler.warmup_ratio * num_steps),
                )
                self.optimizer = [self.optimizer]
                self.scheduler = [self.scheduler]

            elif task in {'pretrain', 'cnndm', 'msqg'}:
                enc_params = [p for n, p in self.model.named_parameters() if n.startswith('encoder')]
                dec_params = [p for n, p in self.model.named_parameters() if not n.startswith('encoder')]
                enc_optimizer = get_optimizer(
                    self.cfg.optimizer.type,
                    enc_params,
                    self.cfg.optimizer.lr_enc,
                    weight_decay=self.cfg.optimizer.weight_decay,
                )
                dec_optimizer = get_optimizer(
                    self.cfg.optimizer.type,
                    dec_params,
                    self.cfg.optimizer.lr_dec,
                    weight_decay=self.cfg.optimizer.weight_decay,
                )
                enc_scheduler = get_scheduler(
                    self.cfg.scheduler.type,
                    enc_optimizer,
                    total_steps=num_steps,
                    warmup_steps=int(self.cfg.scheduler.warmup_ratio_enc * num_steps),
                )
                dec_scheduler = get_scheduler(
                    self.cfg.scheduler.type,
                    dec_optimizer,
                    total_steps=num_steps,
                    warmup_steps=int(self.cfg.scheduler.warmup_ratio_dec * num_steps),
                )
                self.optimizer = [enc_optimizer, dec_optimizer]
                self.scheduler = [enc_scheduler, dec_scheduler]
                self.padding_idx = self.tokenizer.pad_token_id

            self.grad_clip = self.cfg.optimizer.grad_clip
            self.grad_accum_count = self.cfg.trainer.grad_accum_count

        elif self.cfg.backend == 'tensorflow':
            self.trainable_para_names = self.model.trainable_variables()
            self.criterion = None
            self.regularizer = None
            self.optimizer = None
            self.grad_clip = None

        self.mode = list()
        self.cur_data_splits_used_by_routine = list()

        # Process training data
        if self.train_data is not None or self.train_loader is not None:
            # Calculate the number of update steps during training given the local_update_steps
            num_train_batch, num_train_batch_last_epoch, num_train_epoch, num_total_train_batch = \
                self.pre_calculate_batch_epoch_num(self.cfg.federate.local_update_steps)

            self.num_train_epoch = num_train_epoch
            self.num_train_batch = num_train_batch
            self.num_train_batch_last_epoch = num_train_batch_last_epoch
            self.num_total_train_batch = num_total_train_batch

        # Process evaluation data
        for mode in ["val", "test"]:
            setattr(self, "num_{}_epoch".format(mode), 1)
            if self.get("{}_data".format(mode)) is not None or self.get("{}_loader".format(mode)) is not None:
                setattr(
                    self, "num_{}_batch".format(mode),
                    getattr(self, "num_{}_data".format(mode)) //
                    self.cfg.data.batch_size +
                    int(not self.cfg.data.drop_last and bool(
                        getattr(self, "num_{}_data".format(mode)) %
                        self.cfg.data.batch_size)))

    def pre_calculate_batch_epoch_num(self, local_update_steps):
        if self.cfg.trainer.train_steps is not None:
            num_train_batch = self.cfg.trainer.train_steps * self.cfg.trainer.grad_accum_count
            local_update_steps = 1
        else:
            num_train_batch = self.num_train_data // self.cfg.data.batch_size + int(
                not self.cfg.data.drop_last
                and bool(self.num_train_data % self.cfg.data.batch_size))

        if self.cfg.federate.batch_or_epoch == "epoch":
            num_train_epoch = local_update_steps
            num_train_batch_last_epoch = num_train_batch
            num_total_train_batch = local_update_steps * num_train_batch
        else:
            num_train_epoch = math.ceil(local_update_steps / num_train_batch)
            num_train_batch_last_epoch = local_update_steps % num_train_batch
            num_total_train_batch = local_update_steps

        return num_train_batch, num_train_batch_last_epoch, num_train_epoch, num_total_train_batch


class PFedNLPContext(FedNLPContext):
    def setup_vars(self):
        if self.cfg.backend == 'torch':
            self.trainable_para_names = get_trainable_para_names(self.model)
            self.criterion = get_criterion(self.cfg.criterion.type, self.device)
            self.regularizer = get_regularizer(self.cfg.regularizer.type)
            self.tokenizer = setup_tokenizer(self.cfg.model.bert_type)
            self.eval_metrics = None

            task = self.cfg.data.task
            if self.cfg.trainer.train_steps is not None:
                num_steps = self.cfg.trainer.train_steps * self.cfg.federate.total_round_num
            else:
                num_steps = len(self.train_loader) * self.cfg.federate.local_update_steps * \
                            self.cfg.federate.total_round_num

            if task == 'pretrain':
                self.padding_idx = self.tokenizer.pad_token_id
                self.optimizer = dict()
                self.scheduler = dict()
                self.grad_clip = dict()
                enc_params = [p for n, p in self.model.named_parameters() if n.startswith('encoder')]
                dec_params = [p for n, p in self.model.named_parameters() if not n.startswith('encoder')]
                for subtask in self.cfg.data.pretrain_tasks:
                    cfg_optimizer = getattr(self.cfg.optimizer, subtask)
                    cfg_scheduler = getattr(self.cfg.scheduler, subtask)

                    enc_optimizer = get_optimizer(
                        cfg_optimizer.type,
                        enc_params,
                        cfg_optimizer.lr_enc,
                        weight_decay=cfg_optimizer.weight_decay,
                    )
                    dec_optimizer = get_optimizer(
                        cfg_optimizer.type,
                        dec_params,
                        cfg_optimizer.lr_dec,
                        weight_decay=cfg_optimizer.weight_decay,
                    )
                    enc_scheduler = get_scheduler(
                        cfg_scheduler.type,
                        enc_optimizer,
                        total_steps=num_steps,
                        warmup_steps=int(cfg_scheduler.warmup_ratio_enc * num_steps),
                    )
                    dec_scheduler = get_scheduler(
                        cfg_scheduler.type,
                        dec_optimizer,
                        total_steps=num_steps,
                        warmup_steps=int(cfg_scheduler.warmup_ratio_dec * num_steps),
                    )

                    optimizer = [enc_optimizer, dec_optimizer]
                    scheduler = [enc_scheduler, dec_scheduler]
                    self.optimizer[subtask] = optimizer
                    self.scheduler[subtask] = scheduler
                    self.grad_clip[subtask] = cfg_optimizer.grad_clip

            elif task in {'imdb', 'agnews', 'squad', 'newsqa'}:
                self.optimizer = get_optimizer(
                    self.cfg.optimizer.type,
                    self.model,
                    self.cfg.optimizer.lr,
                    weight_decay=self.cfg.optimizer.weight_decay,
                )
                self.scheduler = get_scheduler(
                    self.cfg.scheduler.type,
                    self.optimizer,
                    total_steps=num_steps,
                    warmup_steps=int(self.cfg.scheduler.warmup_ratio * num_steps),
                )
                self.optimizer = [self.optimizer]
                self.scheduler = [self.scheduler]
                self.grad_clip = self.cfg.optimizer.grad_clip

            elif task in {'cnndm', 'msqg'}:
                enc_params = [p for n, p in self.model.named_parameters() if n.startswith('encoder')]
                dec_params = [p for n, p in self.model.named_parameters() if not n.startswith('encoder')]
                enc_optimizer = get_optimizer(
                    self.cfg.optimizer.type,
                    enc_params,
                    self.cfg.optimizer.lr_enc,
                    weight_decay=self.cfg.optimizer.weight_decay,
                )
                dec_optimizer = get_optimizer(
                    self.cfg.optimizer.type,
                    dec_params,
                    self.cfg.optimizer.lr_dec,
                    weight_decay=self.cfg.optimizer.weight_decay,
                )
                enc_scheduler = get_scheduler(
                    self.cfg.scheduler.type,
                    enc_optimizer,
                    total_steps=num_steps,
                    warmup_steps=int(self.cfg.scheduler.warmup_ratio_enc * num_steps),
                )
                dec_scheduler = get_scheduler(
                    self.cfg.scheduler.type,
                    dec_optimizer,
                    total_steps=num_steps,
                    warmup_steps=int(self.cfg.scheduler.warmup_ratio_dec * num_steps),
                )
                self.optimizer = [enc_optimizer, dec_optimizer]
                self.scheduler = [enc_scheduler, dec_scheduler]
                self.padding_idx = self.tokenizer.pad_token_id
                self.grad_clip = self.cfg.optimizer.grad_clip

            self.grad_accum_count = self.cfg.trainer.grad_accum_count

        elif self.cfg.backend == 'tensorflow':
            self.trainable_para_names = self.model.trainable_variables()
            self.criterion = None
            self.regularizer = None
            self.optimizer = None
            self.grad_clip = None

        self.mode = list()
        self.cur_data_splits_used_by_routine = list()

        # Process training data
        if self.train_data is not None or self.train_loader is not None:
            # Calculate the number of update steps during training given the local_update_steps
            num_train_batch, num_train_batch_last_epoch, num_train_epoch, num_total_train_batch = \
                self.pre_calculate_batch_epoch_num(self.cfg.federate.local_update_steps)

            self.num_train_epoch = num_train_epoch
            self.num_train_batch = num_train_batch
            self.num_train_batch_last_epoch = num_train_batch_last_epoch
            self.num_total_train_batch = num_total_train_batch

        # Process evaluation data
        for mode in ["val", "test"]:
            setattr(self, "num_{}_epoch".format(mode), 1)
            if self.get("{}_data".format(mode)) is not None or self.get("{}_loader".format(mode)) is not None:
                setattr(
                    self, "num_{}_batch".format(mode),
                    getattr(self, "num_{}_data".format(mode)) //
                    self.cfg.data.batch_size +
                    int(not self.cfg.data.drop_last and bool(
                        getattr(self, "num_{}_data".format(mode)) %
                        self.cfg.data.batch_size)))
