import math
from torch.optim.lr_scheduler import LambdaLR
from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)


def get_cosine_then_constant_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
):
    """Linear warmup -> cosine decay to ``min_lr_ratio`` by ``num_decay_steps``
    -> constant at ``min_lr_ratio`` for every step after that.

    Unlike the plain ``cosine`` schedule (whose shape depends on the full
    training length and which rises again past ``num_training_steps``), this
    holds a small constant LR once decay finishes, so a run can be extended /
    resumed past ``num_decay_steps`` without the LR moving. It is purely a
    function of the global step, so resuming with ``last_epoch=global_step-1``
    reproduces the schedule exactly.

    factor(step) is multiplied onto the optimizer's base LR:
      step < num_warmup_steps : linear 0 -> 1
      warmup <= step < decay  : cosine 1 -> min_lr_ratio
      step >= num_decay_steps : min_lr_ratio  (constant)
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step >= num_decay_steps:
            return min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_decay_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)
