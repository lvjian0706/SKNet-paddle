import copy
import paddle
import numpy as np
from paddle.io import DistributedBatchSampler, BatchSampler, DataLoader
from ppcls.utils import logger

from ppcls.data import dataloader
# dataset
from ppcls.data.dataloader.imagenet_dataset import ImageNetDataset

# sampler
from ppcls.data import preprocess
from ppcls.data.preprocess import transform


def create_operators(params):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    ops = []
    for operator in params:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = getattr(preprocess, op_name)(**param)
        ops.append(op)

    return ops


def build_dataloader(config, mode, device):
    assert mode in [
        'Train', 'Eval', 'Test'
    ], "Dataset mode should be Train, Eval, Test"
    # build dataset
    config_dataset = config[mode]['dataset']
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop('name')

    dataset = eval(dataset_name)(**config_dataset)

    logger.debug("build dataset({}) success...".format(dataset))

    # build sampler
    config_sampler = config[mode]['sampler']
    if "name" not in config_sampler:
        batch_sampler = None
        batch_size = config_sampler["batch_size"]
        drop_last = config_sampler["drop_last"]
        shuffle = config_sampler["shuffle"]
    else:
        sampler_name = config_sampler.pop("name")
        batch_sampler = eval(sampler_name)(dataset, **config_sampler)

    logger.debug("build batch_sampler({}) success...".format(batch_sampler))

    # build dataloader
    config_loader = config[mode]['loader']
    num_workers = config_loader["num_workers"]
    use_shared_memory = config_loader["use_shared_memory"]

    if batch_sampler is None:
        data_loader = DataLoader(
            dataset=dataset,
            places=device,
            num_workers=num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=None)
    else:
        data_loader = DataLoader(
            dataset=dataset,
            places=device,
            num_workers=num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory,
            batch_sampler=batch_sampler,
            collate_fn=None)

    logger.debug("build data_loader({}) success...".format(data_loader))
    return data_loader
