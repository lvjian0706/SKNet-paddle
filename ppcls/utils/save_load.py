from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import re
import shutil
import tempfile

import paddle
from ppcls.utils import logger
from .download import get_weights_path_from_url

__all__ = ['init_model', 'save_model', 'load_dygraph_pretrain']


def _mkdir_if_not_exist(path):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(param_state_dict)
    return


def load_dygraph_pretrain_from_url(model, pretrained_url, use_ssld=False):
    if use_ssld:
        pretrained_url = pretrained_url.replace("_pretrained",
                                                "_ssld_pretrained")
    local_weight_path = get_weights_path_from_url(pretrained_url).replace(
        ".pdparams", "")
    load_dygraph_pretrain(model, path=local_weight_path)
    return


def init_model(config, net, optimizer=None):
    """
    load model from checkpoint or pretrained_model
    """
    checkpoints = config.get('checkpoints')
    if checkpoints and optimizer is not None:
        assert os.path.exists(checkpoints + ".pdparams"), \
            "Given dir {}.pdparams not exist.".format(checkpoints)
        assert os.path.exists(checkpoints + ".pdopt"), \
            "Given dir {}.pdopt not exist.".format(checkpoints)
        para_dict = paddle.load(checkpoints + ".pdparams")
        opti_dict0 = paddle.load(checkpoints + "_0.pdopt")
        opti_dict1 = paddle.load(checkpoints + "_1.pdopt")
        metric_dict = paddle.load(checkpoints + ".pdstates")
        net.set_dict(para_dict)
        optimizer[0].set_state_dict(opti_dict0)
        optimizer[1].set_state_dict(opti_dict1)
        logger.info("Finish load checkpoints from {}".format(checkpoints))
        return metric_dict

    pretrained_model = config.get('pretrained_model')
    if pretrained_model:
        load_dygraph_pretrain(net, path=pretrained_model)
        logger.info(
            logger.coloring("Finish load pretrained model from {}".format(
                pretrained_model), "HEADER"))


def save_model(net,
               optimizer,
               metric_info,
               model_path,
               model_name="",
               prefix='ppcls'):
    """
    save model to the target path
    """
    if paddle.distributed.get_rank() != 0:
        return
    model_path = os.path.join(model_path, model_name)
    _mkdir_if_not_exist(model_path)
    model_path = os.path.join(model_path, prefix)

    paddle.save(net.state_dict(), model_path + ".pdparams")
    paddle.save(optimizer[0].state_dict(), model_path + "_0.pdopt")
    paddle.save(optimizer[1].state_dict(), model_path + "_1.pdopt")
    paddle.save(metric_info, model_path + ".pdstates")
    logger.info("Already save model in {}".format(model_path))
