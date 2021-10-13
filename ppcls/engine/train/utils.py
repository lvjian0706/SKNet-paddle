from __future__ import absolute_import, division, print_function

import datetime
from ppcls.utils import logger
from ppcls.utils.misc import AverageMeter


def update_metric(trainer, out, batch, batch_size):
    # calc metric
    if trainer.train_metric_func is not None:
        metric_dict = trainer.train_metric_func(out, batch[-1])
        for key in metric_dict:
            if key not in trainer.output_info:
                trainer.output_info[key] = AverageMeter(key, '7.5f')
            trainer.output_info[key].update(metric_dict[key].numpy()[0],
                                            batch_size)


def update_loss(trainer, loss_dict, batch_size):
    # update_output_info
    for key in loss_dict:
        if key not in trainer.output_info:
            trainer.output_info[key] = AverageMeter(key, '7.5f')
        trainer.output_info[key].update(loss_dict[key].numpy()[0], batch_size)


def log_info(trainer, batch_size, epoch_id, iter_id):
    lr_msg = "lr: {:.5f}".format(trainer.lr_sch.get_lr())
    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, trainer.output_info[key].val)
        if key=='CELoss' else "{}: {:.5f}".format(key, trainer.output_info[key].avg) for key in trainer.output_info
    ])
    time_msg = "s, ".join([
        "{}: {:.5f}".format(key, trainer.time_info[key].avg)
        for key in trainer.time_info
    ])

    ips_msg = "ips: {:.5f} images/sec".format(
        batch_size / trainer.time_info["batch_cost"].avg)
    eta_sec = ((trainer.config["Global"]["epochs"] - epoch_id + 1
                ) * len(trainer.train_dataloader) - iter_id
               ) * trainer.time_info["batch_cost"].avg
    eta_msg = "eta: {:s}".format(str(datetime.timedelta(seconds=int(eta_sec))))
    logger.info("[Train][Epoch {}/{}][Iter: {}/{}]{}, {}, {}, {}, {}".format(
        epoch_id, trainer.config["Global"]["epochs"], iter_id,
        len(trainer.train_dataloader), lr_msg, metric_msg, time_msg, ips_msg,
        eta_msg))

    logger.scaler(
        name="lr",
        value=trainer.lr_sch.get_lr(),
        step=trainer.global_step,
        writer=trainer.vdl_writer)
    for key in trainer.output_info:
        logger.scaler(
            name="train_{}".format(key),
            value=trainer.output_info[key].avg,
            step=trainer.global_step,
            writer=trainer.vdl_writer)
