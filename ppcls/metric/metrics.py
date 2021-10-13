import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TopkAcc(nn.Layer):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        metric_dict = dict()
        for k in self.topk:
            metric_dict["top{}".format(k)] = paddle.metric.accuracy(
                x, label, k=k)
        return metric_dict