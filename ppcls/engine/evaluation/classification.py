from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import platform
import paddle

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger


def classification_eval(engine, epoch_id=0):
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    metric_key = None
    tic = time.time()
    max_iter = len(engine.eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(engine.eval_dataloader)
    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        batch[0] = paddle.to_tensor(batch[0]).astype("float32")
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")
        # image input
        out = engine.model(batch[0])
        # calc loss
        if engine.eval_loss_func is not None:
            loss_dict = engine.eval_loss_func(out, batch[1])
            for key in loss_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')
                output_info[key].update(loss_dict[key].numpy()[0], batch_size)
        # calc metric
        if engine.eval_metric_func is not None:
            metric_dict = engine.eval_metric_func(out, batch[1])
            if paddle.distributed.get_world_size() > 1:
                for key in metric_dict:
                    paddle.distributed.all_reduce(
                        metric_dict[key], op=paddle.distributed.ReduceOp.SUM)
                    metric_dict[key] = metric_dict[
                        key] / paddle.distributed.get_world_size()
            for key in metric_dict:
                if metric_key is None:
                    metric_key = key
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')

                output_info[key].update(metric_dict[key].numpy()[0],
                                        batch_size)

        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].val)
                for key in output_info
            ])
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()
    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, output_info[key].avg) for key in output_info
    ])
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    # do not try to save best eval.model
    if engine.eval_metric_func is None:
        return -1
    # return 1st metric in the dict
    return output_info[metric_key].avg
