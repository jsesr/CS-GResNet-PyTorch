import torch.nn as nn
from Args.args import ARGS
import numpy as np

# 取每个layer的最后一层的hw
HW_Layer = {
    'layer1': [56, 56, 64],
    'layer2': [28, 28, 128],
    'layer3': [14, 14, 256],
    'layer4': [7, 7, 512],
} if (ARGS.model == 'GCN' and (not ARGS.GCN_is_maxpool)) else {
    'layer1': [28, 28, 64],
    'layer2': [14, 14, 128],
    'layer3': [7, 7, 256],
    'layer4': [4, 4, 512],
}


class ChannelShift(nn.Module):
    def __init__(self, net, h, w, input_channels, n_div=8, strategy='start'):
        super(ChannelShift, self).__init__()
        self.net = net
        M = 4 if ARGS.model == 'GCN' else 1
        self.fold = M * h * w // n_div
        self.conv = nn.Conv1d(in_channels=M * h * w, out_channels=M * h * w, kernel_size=3,
                              padding=1, groups=M * h * w, bias=False)

        self.init_conv(n_div, M * h * w, strategy)

    def forward(self, x):
        x = self.enhance(x)
        return self.net(x)

    # TODO:NOT TEST!!!
    def init_conv(self, n_div, hw, strategy):
        self.conv.weight.requires_grad = False
        self.conv.weight.data.zero_()

        if strategy == 'start':
            i = 0
        elif strategy == 'middle':
            i = int(n_div // 2 - 1)
        elif strategy == 'end':
            i = n_div - 2
        elif strategy == 'stochastic':
            # 从hw中随机选取2*self.fold个数
            i = np.random.choice(np.arange(hw), 2 * self.fold, replace=False)
            self.init_conv_stochastic(i, hw)
            return
        else:
            assert False, 'init_conv()!!!'

        # init_conv for the strategies except for the stochastic
        self.conv.weight.data[i * self.fold:(i + 1) * self.fold, 0, 2] = 1  # shift left
        self.conv.weight.data[(i + 1) * self.fold: (i + 2) * self.fold, 0, 0] = 1  # shift right
        if i * self.fold != 0:
            self.conv.weight.data[0:i * self.fold, 0, 1] = 1  # fixed
        if (i + 2) * self.fold < hw:
            self.conv.weight.data[(i + 2) * self.fold:, 0, 1] = 1  # fixed

    def init_conv_stochastic(self, i, hw):
        self.conv.weight.data[i[:self.fold], 0, 2] = 1  # shift left
        self.conv.weight.data[i[self.fold:], 0, 0] = 1  # shift right
        self.conv.weight.data[[k for k in set(range(hw)) - set(i)], 0, 1] = 1  # fixed

    def enhance(self, x):
        bs, c, h, w = x.size()
        x = x.reshape(bs, c // 4, 4 * h * w) if ARGS.model == 'GCN' else x.reshape(bs, c, h * w)
        # (bs, 4*h*w, c//4)/(bs, h*w, c)
        x = x.permute([0, 2, 1])
        x = self.conv(x)
        # (bs, c//4, 4*h*w)/(bs, c, h*w)
        x = x.permute([0, 2, 1])
        x = x.reshape(bs, c, h, w)
        return x


def make_channel_shift(net):
    assert ARGS.base_model in ['resnet18', 'resnet34'], 'make_temporal_shift()!!!'

    # args.TSM_position: ['layer1'] or ['layer1', 'layer3']
    for position in ARGS.TSM_position:
        layer = getattr(net, position)
        layer = make_BasicBlock_shift(layer, HW_Layer[position], ARGS.TSM_div)
        setattr(net, position, layer)

    return net


def make_BasicBlock_shift(stage, hwc, n_div):
    # 在每一个blocks的最后一个conv进行channel shift
    blocks = list(stage.children())
    if ARGS.TSM_channel_enhance == 1:
        blocks[-1] = ChannelShift(blocks[-1], n_div=n_div)
    elif ARGS.TSM_channel_enhance == 2:
        blocks[-1] = ChannelShift(blocks[-1], hwc[0], hwc[1], hwc[2], n_div=n_div)
    elif ARGS.TSM_channel_enhance == 3:
        blocks[-1] = ChannelShift(blocks[-1], hwc[0], hwc[1], hwc[2],
                                  n_div=n_div, strategy=ARGS.channel_shift_strategy)
    else:
        assert False, 'make_BasicBlock_shift()'

    return nn.Sequential(*blocks)


if __name__ == '__main__':
    pass
