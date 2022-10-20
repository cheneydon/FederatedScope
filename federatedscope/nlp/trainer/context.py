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
            self.tokenizer = setup_tokenizer(self.cfg)
            self.eval_metrics = None

            if self.cfg.trainer.train_steps is not None:
                num_steps = self.cfg.trainer.train_steps * self.cfg.federate.total_round_num
            else:
                num_steps = len(self.train_loader) * self.cfg.federate.local_update_steps * \
                            self.cfg.federate.total_round_num

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
            self.grad_clip = self.cfg.optimizer.grad_clip
            self.grad_accum_count = self.cfg.trainer.grad_accum_count
            self.padding_idx = self.tokenizer.pad_token_id

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


# class PFedNLPContext(FedNLPContext):
#     def setup_vars(self):
#         if self.cfg.backend == 'torch':
#             self.trainable_para_names = get_trainable_para_names(self.model)
#             self.criterion = get_criterion(self.cfg.criterion.type, self.device)
#             self.regularizer = get_regularizer(self.cfg.regularizer.type)
#             self.tokenizer = setup_tokenizer(self.cfg)
#             self.eval_metrics = None
#
#             if self.cfg.trainer.train_steps is not None:
#                 num_steps = self.cfg.trainer.train_steps * self.cfg.federate.total_round_num
#             else:
#                 num_steps = len(self.train_loader) * self.cfg.federate.local_update_steps * \
#                             self.cfg.federate.total_round_num
#
#             self.optimizer = get_optimizer(
#                 self.cfg.optimizer.type,
#                 self.model,
#                 self.cfg.optimizer.lr,
#                 weight_decay=self.cfg.optimizer.weight_decay,
#             )
#             self.scheduler = get_scheduler(
#                 self.cfg.scheduler.type,
#                 self.optimizer,
#                 total_steps=num_steps,
#                 warmup_steps=int(self.cfg.scheduler.warmup_ratio * num_steps),
#             )
#             self.grad_clip = self.cfg.optimizer.grad_clip
#             self.grad_accum_count = self.cfg.trainer.grad_accum_count
#             self.padding_idx = self.tokenizer.pad_token_id
#
#         elif self.cfg.backend == 'tensorflow':
#             self.trainable_para_names = self.model.trainable_variables()
#             self.criterion = None
#             self.regularizer = None
#             self.optimizer = None
#             self.grad_clip = None
#
#         self.mode = list()
#         self.cur_data_splits_used_by_routine = list()
#
#         # Process training data
#         if self.train_data is not None or self.train_loader is not None:
#             # Calculate the number of update steps during training given the local_update_steps
#             num_train_batch, num_train_batch_last_epoch, num_train_epoch, num_total_train_batch = \
#                 self.pre_calculate_batch_epoch_num(self.cfg.federate.local_update_steps)
#
#             self.num_train_epoch = num_train_epoch
#             self.num_train_batch = num_train_batch
#             self.num_train_batch_last_epoch = num_train_batch_last_epoch
#             self.num_total_train_batch = num_total_train_batch
#
#         # Process evaluation data
#         for mode in ["val", "test"]:
#             setattr(self, "num_{}_epoch".format(mode), 1)
#             if self.get("{}_data".format(mode)) is not None or self.get("{}_loader".format(mode)) is not None:
#                 setattr(
#                     self, "num_{}_batch".format(mode),
#                     getattr(self, "num_{}_data".format(mode)) //
#                     self.cfg.data.batch_size +
#                     int(not self.cfg.data.drop_last and bool(
#                         getattr(self, "num_{}_data".format(mode)) %
#                         self.cfg.data.batch_size)))
