from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import optimizer as optim


def add_weight_decay(model_list, weight_decay=1e-4):
    decay = []
    no_decay = []
    for model in model_list:
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class Momentum(object):
    """
    Simple Momentum optimizer with velocity state.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum,
                 weight_decay=None,
                 grad_clip=None,
                 multi_precision=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.multi_precision = multi_precision

    def __call__(self, model_list):
        #parameters = sum([m.parameters() for m in model_list], [])
        parameters = add_weight_decay(model_list, self.weight_decay)
        no_decay_param = parameters[0]['params']
        decay_param = parameters[1]['params']
        #import pdb
        #pdb.set_trace()
        #self.weight_decay = 0.
        opt1 = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            use_nesterov=True,
            weight_decay=0.,
            grad_clip=self.grad_clip,
            #multi_precision=self.multi_precision,
            parameters=no_decay_param)
        opt2 = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            use_nesterov=True,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            #multi_precision=self.multi_precision,
            parameters=decay_param)
        #import pdb
        #pdb.set_trace()
        return (opt1,opt2)
