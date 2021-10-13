from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from paddle.optimizer import lr
from paddle.optimizer.lr import LRScheduler

from ppcls.utils import logger


class Step(object):
    """
    Piecewise learning rate decay
    Args:
        step_each_epoch(int): steps each epoch
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_size,
                 step_each_epoch,
                 epochs,
                 gamma,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.step_size = step_each_epoch * step_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_steps = round(warmup_epoch * step_each_epoch)
        self.warmup_start_lr = warmup_start_lr

    def __call__(self):
        learning_rate = lr.StepDecay(
            learning_rate=self.learning_rate,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=self.last_epoch)
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate
