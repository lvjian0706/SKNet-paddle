from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import paddle

from ppcls.utils import logger

from . import optimizer

__all__ = ['build_optimizer']


def build_lr_scheduler(lr_config, epochs, step_each_epoch):
    from . import learning_rate
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    if 'name' in lr_config:
        lr_name = lr_config.pop('name')
        lr = getattr(learning_rate, lr_name)(**lr_config)
        if isinstance(lr, paddle.optimizer.lr.LRScheduler):
            return lr
        else:
            return lr()
    else:
        lr = lr_config['learning_rate']
    return lr


def build_optimizer(config, epochs, step_each_epoch, model_list):
    config = copy.deepcopy(config)
    # step1 build lr
    lr = build_lr_scheduler(config.pop('lr'), epochs, step_each_epoch)
    logger.debug("build lr ({}) success..".format(lr))
    # step2 build regularization
    if 'regularizer' in config and config['regularizer'] is not None:
        if 'weight_decay' in config:
            logger.warning(
                "ConfigError: Only one of regularizer and weight_decay can be set in Optimizer Config. \"weight_decay\" has been ignored."
            )
        reg_config = config.pop('regularizer')
        reg_name = reg_config.pop('name') + 'Decay'
        reg = getattr(paddle.regularizer, reg_name)(**reg_config)
        config["weight_decay"] = reg
        logger.debug("build regularizer ({}) success..".format(reg))
    # step3 build optimizer
    optim_name = config.pop('name')
    if 'clip_norm' in config:
        clip_norm = config.pop('clip_norm')
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    else:
        grad_clip = None
    optim = getattr(optimizer, optim_name)(learning_rate=lr,
                                           grad_clip=grad_clip,
                                           **config)(model_list=model_list)
    logger.debug("build optimizer ({}) success..".format(optim))
    return optim, lr
