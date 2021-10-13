import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
from paddle.nn.initializer import  Constant
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url


zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def downsample_conv(in_channels, out_channels, kernel_size, stride=1,
                    dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2D
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2D(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias_attr=False),
        norm_layer(out_channels)
    ])


class ConvBnAct(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bias=False, norm_layer=nn.BatchNorm2D, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()

        padding = get_padding(kernel_size, stride, dilation)

        self.conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias_attr=False)

        self.bn = norm_layer(out_channels)
        self.act = act_layer() if act_layer is not None else None

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SelectiveKernelAttn(nn.Layer):
    def __init__(self, channels, num_paths=2, attn_channels=32,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2D):
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.fc_reduce = nn.Conv2D(channels, attn_channels, kernel_size=1, bias_attr=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer()
        self.fc_select = nn.Conv2D(attn_channels, channels * num_paths, kernel_size=1, bias_attr=False)

    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = x.sum(axis=1).mean(axis=(2, 3), keepdim=True)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        x = x.reshape([B, self.num_paths, C // self.num_paths, H, W])
        x = F.softmax(x, axis=1)
        return x


class SelectiveKernel(nn.Layer):
    def __init__(self, in_channels, out_channels=None, kernel_size=None, stride=1, dilation=1, groups=1,
                 rd_ratio=1./16, rd_channels=None, rd_divisor=8, keep_3x3=True,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2D):

        super(SelectiveKernel, self).__init__()
        out_channels = out_channels or in_channels
        kernel_size = kernel_size or [3, 5]  # default to one 3x3 and one 5x5 branch. 5x5 -> 3x3 + dilation
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        groups = min(out_channels, groups)

        conv_kwargs = dict(
            stride=stride, groups=groups, act_layer=act_layer, norm_layer=norm_layer)
        self.paths = nn.LayerList([
            ConvBnAct(in_channels, out_channels, kernel_size=k, dilation=d, **conv_kwargs)
            for k, d in zip(kernel_size, dilation)])

        attn_channels = rd_channels or make_divisible(out_channels * rd_ratio, divisor=rd_divisor)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

    def forward(self, x):
        x_paths = [op(x) for op in self.paths]
        x = paddle.stack(x_paths, axis=1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = paddle.sum(x, axis=1)
        return x


class SelectiveKernelBottleneck(nn.Layer):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, sk_kwargs=None, reduce_first=1, dilation=1, first_dilation=None,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2D):
        super(SelectiveKernelBottleneck, self).__init__()

        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = ConvBnAct(inplanes, first_planes, kernel_size=1, **conv_kwargs)
        self.conv2 = SelectiveKernel(
            first_planes, width, stride=stride, dilation=first_dilation, groups=cardinality,
            **conv_kwargs, **sk_kwargs)
        conv_kwargs['act_layer'] = None
        self.conv3 = ConvBnAct(width, outplanes, kernel_size=1, **conv_kwargs)
        self.act = act_layer()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act(x)
        return x


def make_blocks(block_fn, channels, block_repeats, inplanes, reduce_first=1,
                output_stride=32, down_kernel_size=1, **kwargs):
    stages = []
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks) in enumerate(zip(channels, block_repeats)):
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append(nn.Sequential(*blocks))

    return stages


class SKNet(nn.Layer):
    def __init__(self, block, layers, num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64, output_stride=32,
                 block_reduce_first=1, down_kernel_size=1, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2D,  block_args=None):
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        super(SKNet, self).__init__()

        # Stem
        inplanes = 64
        self.conv1 = nn.Conv2D(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer()
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, **block_args)

        self.stage_modules = stage_modules

        for i, stage in enumerate(self.stage_modules):
            self.add_sublayer('layer%d' % (i + 1), stage)  # layer1, layer2, etc

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool = nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(channels[-1] * 2 * 1.0)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(self.num_features, num_classes, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))
        #self.fc = nn.Linear(self.num_features, num_classes)
        self.init_weights()

    def init_weights(self, zero_init_last_bn=False):
        for n, m in self.named_sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                ones_(m.weight)
                zeros_(m.bias)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        for stage in self.stage_modules:
            x = stage(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def SKNet50(pretrained=False, use_ssld=False, **kwargs):
    sk_kwargs = dict(rd_ratio=1/16, rd_divisor=32)
    model = SKNet(block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3],
                  num_classes=1000, cardinality=32, base_width=4,
                  block_args=dict(sk_kwargs=sk_kwargs))
    _load_pretrained(
        pretrained, model, None, use_ssld=use_ssld)
    return model