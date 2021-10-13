import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CELoss(nn.Layer):
    """
    Cross entropy loss
    """

    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon

    def _labelsmoothing(self, target, class_num):
        if len(target.shape) == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            x = -F.log_softmax(x, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
        else:
            if label.shape[-1] == x.shape[-1]:
                label = F.softmax(label, axis=-1)
                soft_label = True
            else:
                soft_label = False
            loss = F.cross_entropy(x, label=label, soft_label=soft_label)
        loss = loss.mean()
        return {"CELoss": loss}
