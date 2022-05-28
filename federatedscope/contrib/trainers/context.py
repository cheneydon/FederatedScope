import math
import logging
from federatedscope.core.trainers.context import Context
from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.model_builder import get_trainable_para_names
from federatedscope.core.auxiliaries.regularizer_builder import get_regularizer
from federatedscope.contrib.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.contrib.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.contrib.auxiliaries.loss_builder import abs_loss
from federatedscope.contrib.metrics.generation.predictor import build_predictor

logger = logging.getLogger(__name__)


class MyContext(Context):
    def setup_vars(self):
        if self.cfg.backend == 'torch':
            self.trainable_para_names = get_trainable_para_names(self.model)
            self.criterion = get_criterion(self.cfg.criterion.type, self.device)
            self.regularizer = get_regularizer(self.cfg.regularizer.type)
            self.val_metrics = None
            self.test_metrics = None

            if self.cfg.data.type == 'cnndm':
                self.optimizer = get_optimizer(
                    self.cfg.optimizer.type,
                    self.model,
                    [self.cfg.optimizer.lr_enc, self.cfg.optimizer.lr_dec],
                    weight_decay=self.cfg.optimizer.weight_decay,
                    max_grad_norm=self.cfg.optimizer.grad_clip,
                    warmup_steps_enc=self.cfg.optimizer.warmup_steps_enc,
                    warmup_steps_dec=self.cfg.optimizer.warmup_steps_dec)

                self.symbols = {'BOS': self.tokenizer.vocab['[unused0]'], 'EOS': self.tokenizer.vocab['[unused1]'],
                                'PAD': self.tokenizer.pad_token_id, 'EOQ': self.tokenizer.vocab['[unused2]']}
                self.train_loss_func = abs_loss(self.model.generator, self.symbols, self.model.vocab_size,
                                                self.model.bert.device, train=True, label_smoothing=self.cfg.model.label_smoothing)
                self.val_loss_func = abs_loss(self.model.generator, self.symbols, self.model.vocab_size,
                                              self.model.bert.device, train=False)
                self.test_loss_func = self.val_loss_func
                self.predictor = build_predictor(self.cfg.test, self.tokenizer, self.symbols, self.model, logger)

            else:
                self.optimizer = get_optimizer(
                    self.cfg.optimizer.type,
                    self.model,
                    self.cfg.optimizer.lr,
                    weight_decay=self.cfg.optimizer.weight_decay)
                self.grad_clip = self.cfg.optimizer.grad_clip

                num_steps = len(self.train_loader) * self.cfg.federate.local_update_steps * \
                            self.cfg.federate.total_round_num
                self.scheduler = get_scheduler(
                    self.cfg.scheduler.type,
                    self.optimizer,
                    total_steps=num_steps,
                    warmup_steps=int(self.cfg.scheduler.warmup_ratio * num_steps))

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
            num_train_batch, num_train_batch_last_epoch, num_train_epoch, num_total_train_batch = self.pre_calculate_batch_epoch_num(
                self.cfg.federate.local_update_steps)

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
        num_train_batch = self.num_train_data // self.cfg.data.batch_size + int(
            not self.cfg.data.drop_last
            and bool(self.num_train_data % self.cfg.data.batch_size))
        if self.cfg.trainer.train_steps is not None:
            num_train_batch = self.cfg.trainer.train_steps * self.cfg.trainer.grad_accum_count
        if self.cfg.federate.batch_or_epoch == "epoch":
            num_train_epoch = local_update_steps
            num_train_batch_last_epoch = num_train_batch
            num_total_train_batch = local_update_steps * num_train_batch
        else:
            num_train_epoch = math.ceil(local_update_steps / num_train_batch)
            num_train_batch_last_epoch = local_update_steps % num_train_batch
            num_total_train_batch = local_update_steps
        return num_train_batch, num_train_batch_last_epoch, num_train_epoch, num_total_train_batch
