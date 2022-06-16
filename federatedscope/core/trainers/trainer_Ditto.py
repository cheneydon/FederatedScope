import copy

import torch

# from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
from federatedscope.contrib.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from typing import Type


def wrap_DittoTrainer(
        base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    """
    Build a `DittoTrainer` with a plug-in manner, by registering new functions into specific `BaseTrainer`

    The Ditto implementation, "Ditto: Fair and Robust Federated Learning Through Personalization. (ICML2021)"
    based on the Algorithm 2 in their paper and official codes: https://github.com/litian96/ditto
    """

    # ---------------- attribute-level plug-in -----------------------
    init_Ditto_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------
    base_trainer.register_hook_in_train(
        new_hook=hook_on_fit_start_set_regularized_para,
        trigger="on_fit_start",
        insert_pos=0)
    base_trainer.register_hook_in_train(
        new_hook=hook_on_batch_start_switch_model,
        trigger="on_batch_start",
        insert_pos=0)
    # evaluation is based on the local personalized model
    base_trainer.register_hook_in_eval(
        new_hook=hook_on_fit_start_switch_local_model,
        trigger="on_fit_start",
        insert_pos=0)
    base_trainer.register_hook_in_eval(
        new_hook=hook_on_fit_end_switch_global_model,
        trigger="on_fit_end",
        insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=hook_on_fit_end_free_cuda,
                                        trigger="on_fit_end",
                                        insert_pos=-1)
    base_trainer.register_hook_in_eval(new_hook=hook_on_fit_end_free_cuda,
                                       trigger="on_fit_end",
                                       insert_pos=-1)

    return base_trainer


def init_Ditto_ctx(base_trainer):
    """
    init necessary attributes used in Ditto,
    `global_model` acts as the shared global model in FedAvg;
    `local_model` acts as personalized model will be optimized with regularization based on weights of `global_model`

    """
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    ctx.global_model = copy.deepcopy(ctx.model)
    ctx.local_model = copy.deepcopy(ctx.model)  # the personalized model

    # ctx.optimizer_for_global_model = get_optimizer(
    #     cfg.optimizer.type,
    #     ctx.global_model,
    #     cfg.optimizer.lr,
    #     weight_decay=cfg.optimizer.weight_decay)
    # ctx.optimizer_for_local_model = get_optimizer(
    #     cfg.optimizer.type,
    #     ctx.local_model,
    #     cfg.personalization.lr,
    #     weight_decay=cfg.optimizer.weight_decay)

    if cfg.data.type == 'cnndm':
        ctx.optimizer_for_global_model = get_optimizer(
            cfg.optimizer.type,
            ctx.global_model,
            [cfg.optimizer.lr_enc, cfg.optimizer.lr_dec],
            weight_decay=cfg.optimizer.weight_decay,
            max_grad_norm=cfg.optimizer.grad_clip,
            warmup_steps_enc=int(cfg.optimizer.warmup_steps_enc * 0.25),
            warmup_steps_dec=int(cfg.optimizer.warmup_steps_dec * 0.25))
        ctx.optimizer_for_local_model = get_optimizer(
            cfg.optimizer.type,
            ctx.local_model,
            [cfg.optimizer.lr_enc, cfg.optimizer.lr_dec],
            weight_decay=cfg.optimizer.weight_decay,
            max_grad_norm=cfg.optimizer.grad_clip,
            warmup_steps_enc=int(cfg.optimizer.warmup_steps_enc * 0.75),
            warmup_steps_dec=int(cfg.optimizer.warmup_steps_dec * 0.75))

        ctx.scheduler_for_global_model = None
        ctx.scheduler_for_local_model = None
    else:
        ctx.optimizer_for_global_model = get_optimizer(
            cfg.optimizer.type,
            ctx.global_model,
            cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay)
        ctx.optimizer_for_local_model = get_optimizer(
            cfg.optimizer.type,
            ctx.local_model,
            cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay)

        assert cfg.trainer.train_steps is not None
        num_steps = cfg.trainer.train_steps * cfg.federate.total_round_num

        ctx.scheduler_for_global_model = get_scheduler(
            cfg.scheduler.type,
            ctx.optimizer_for_global_model,
            total_steps=num_steps,
            warmup_steps=int(cfg.scheduler.warmup_ratio * num_steps))
        ctx.scheduler_for_local_model = get_scheduler(
            cfg.scheduler.type,
            ctx.optimizer_for_local_model,
            total_steps=num_steps * 3,
            warmup_steps=int(cfg.scheduler.warmup_ratio * num_steps * 3))

    ctx.optimizer_for_local_model = wrap_regularized_optimizer(
        ctx.optimizer_for_local_model, cfg.personalization.regular_weight)

    ctx.model = ctx.global_model
    del ctx.optimizer

    # track the batch_num, epoch_num, for local & global model respectively
    ctx.num_train_batch_for_local_model, ctx.num_train_batch_last_epoch_for_local_model, \
    ctx.num_train_epoch_for_local_model, ctx.num_total_train_batch = \
        ctx.pre_calculate_batch_epoch_num(cfg.personalization.local_update_steps)

    # In the first `num_train_batch`, `num_train_batch_last_epoch`, and `num_train_epoch`,
    # we will manipulate local models, and manipulate global model in the remaining steps
    ctx.num_train_batch += ctx.num_train_batch_for_local_model
    ctx.num_train_batch_last_epoch += ctx.num_train_batch_last_epoch_for_local_model
    ctx.num_train_epoch += ctx.num_train_epoch_for_local_model


def hook_on_fit_start_set_regularized_para(ctx):
    # set the compared model data for local personalized model
    # ctx.global_model.to(ctx.device)
    # ctx.local_model.to(ctx.device)
    ctx.global_model.train()
    ctx.local_model.train()

    if isinstance(ctx.optimizer_for_local_model, list):
        compared_global_model_para_enc = [{
            "params": [p for n, p in list(ctx.global_model.named_parameters()) if n.startswith('bert')]
        }]
        compared_global_model_para_dec = [{
            "params": [p for n, p in list(ctx.global_model.named_parameters()) if not n.startswith('bert')]
        }]
        ctx.optimizer_for_local_model[0].set_compared_para_group(compared_global_model_para_enc)
        ctx.optimizer_for_local_model[1].set_compared_para_group(compared_global_model_para_dec)
    else:
        compared_global_model_para = [{
            "params": list(ctx.global_model.parameters())
        }]
        ctx.optimizer_for_local_model.set_compared_para_group(
            compared_global_model_para)


def hook_on_batch_start_switch_model(ctx):
    last_epoch_use_local_model = ctx.cur_epoch_i == (ctx.num_train_epoch - 1) and \
                                 ctx.cur_batch_i < ctx.num_train_batch_last_epoch_for_local_model
    use_local_model = last_epoch_use_local_model or ctx.cur_epoch_i < ctx.num_train_epoch_for_local_model or \
                      ctx.cur_batch_i < ctx.num_train_batch_for_local_model

    first_use_local_model = ctx.cur_epoch_i == 0 and ctx.cur_batch_i == 0
    first_use_global_model = ctx.cur_epoch_i == ctx.num_train_epoch_for_local_model and \
                             ctx.cur_batch_i == ctx.num_train_batch_for_local_model
    if first_use_local_model or first_use_global_model:
        ctx.model.to(torch.device("cpu"))
        # torch.cuda.empty_cache()

        if use_local_model:
            ctx.model = ctx.local_model
            ctx.optimizer = ctx.optimizer_for_local_model
            ctx.scheduler = ctx.scheduler_for_local_model
        else:
            ctx.model = ctx.global_model
            ctx.optimizer = ctx.optimizer_for_global_model
            ctx.scheduler = ctx.scheduler_for_global_model

        ctx.model.to(ctx.device)


# Note that Ditto only updates the para of global_model received from other FL participants,
# and in the remaining steps, ctx.model has been = ctx.global_model, thus we do not need register the following hook
# def hook_on_fit_end_link_global_model(ctx):
#     ctx.model = ctx.global_model


def hook_on_fit_start_switch_local_model(ctx):
    ctx.model.to(torch.device("cpu"))
    # torch.cuda.empty_cache()
    ctx.model = ctx.local_model
    ctx.model.eval()


def hook_on_fit_end_switch_global_model(ctx):
    ctx.model.to(torch.device("cpu"))
    # torch.cuda.empty_cache()
    ctx.model = ctx.global_model


def hook_on_fit_end_free_cuda(ctx):
    # ctx.global_model.to(torch.device("cpu"))
    # ctx.local_model.to(torch.device("cpu"))
    ctx.model.to(torch.device("cpu"))
    # torch.cuda.empty_cache()

