import copy
import importlib

import paddle.nn as nn
from paddle.jit import to_static
from paddle.static import InputSpec

from . import backbone
from .backbone import *
from .utils import *
from ppcls.utils import logger

__all__ = ["build_model"]


def build_model(config):
    config = copy.deepcopy(config)
    model_type = config.pop("name")
    mod = importlib.import_module(__name__)
    arch = getattr(mod, model_type)(**config)
    return arch


def apply_to_static(config, model):
    support_to_static = config['Global'].get('to_static', False)

    if support_to_static:
        specs = None
        if 'image_shape' in config['Global']:
            specs = [InputSpec([None] + config['Global']['image_shape'])]
        model = to_static(model, input_spec=specs)
        logger.info("Successfully to apply @to_static with specs: {}".format(
            specs))
    return model