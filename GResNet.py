import torch
import torch.nn as nn
from Models.Gabor.gcn.layers import GConv
import sys

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, M, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return GConv(in_planes // M, out_planes // M, kernel_size=3, stride=stride,
                 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, M, stride=1):
    """1x1 convolution"""
    return GConv(in_planes // M, out_planes // M, kernel_size=1, stride=stride, bias=False)


class Downsample(nn.Module):
    def __init__(self, inplanes, planes, M, GCN_Norm_M, stride=1, norm_layer=None):
        super(Downsample, self).__init__()
        self.GCN_Norm_M = GCN_Norm_M
        self.M = M
        self.conv1x1 = conv1x1(inplanes, planes, M, stride)
        # self.bn = [norm_layer(planes // M).cuda()] * 4 if GCN_Norm_M else norm_layer(planes)
        self.bn = Bn(planes, M, GCN_Norm_M, norm_layer)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        return x


class Bn(nn.Module):
    def __init__(self, planes, M, GCN_Norm_M, norm_layer=None):
        super(Bn, self).__init__()
        self.GCN_Norm_M = GCN_Norm_M
        self.M = M
        # self.bn = [norm_layer(planes // M).cuda()] * 4 if GCN_Norm_M else norm_layer(planes)
        self.bn = norm_layer(planes // M) if GCN_Norm_M else norm_layer(planes)

    def forward(self, x):
        if self.GCN_Norm_M:
            bs, c, h, w = x.shape
            for i in range(self.M):
                start, end = i * c // self.M, (i + 1) * c // self.M
                # x[:, start:end, :, :] = self.bn[i](x[:, start:end, :, :])
                x[:, start:end, :, :] = self.bn(x[:, start:end, :, :])
        else:
            x = self.bn(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, M, GCN_Norm_M, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.GCN_Norm_M = GCN_Norm_M
        self.M = M
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, M, stride)
        # self.bn1 = norm_layer(planes)
        # self.bn1 = [norm_layer(planes // M).cuda()] * 4 if GCN_Norm_M else norm_layer(planes)
        self.bn1 = Bn(planes, M, GCN_Norm_M, norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, M)
        # self.bn2 = norm_layer(planes)
        # self.bn2 = [norm_layer(planes // M).cuda()] * 4 if GCN_Norm_M else norm_layer(planes)
        self.bn2 = Bn(planes, M, GCN_Norm_M, norm_layer)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        # out = bn(out, self.bn1, self.GCN_Norm_M, self.M)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        # out = bn(out, self.bn2, self.GCN_Norm_M, self.M)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, M, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, M)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, M, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, M)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, M, GCN_is_maxpool, GCN_Norm_M, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, input_channel=3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.M = M
        self.GCN_Norm_M = GCN_Norm_M
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = GConv(input_channel, self.inplanes // M, kernel_size=5, stride=2, padding=2,
                           bias=False, expand=True)
        # self.bn1 = [norm_layer(self.inplanes // M).cuda()] * 4 if GCN_Norm_M else norm_layer(self.inplanes)
        self.bn1 = Bn(self.inplanes, M, GCN_Norm_M, norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if GCN_is_maxpool else nn.Identity()
        self.layer1 = self._make_layer(block, 64, layers[0], M, GCN_Norm_M)
        self.layer2 = self._make_layer(block, 128, layers[1], M, GCN_Norm_M, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], M, GCN_Norm_M, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], M, GCN_Norm_M, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion // M, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, M, GCN_Norm_M, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, M, stride),
            #     norm_layer(planes * block.expansion),
            # )
            downsample = Downsample(self.inplanes, planes * block.expansion, M, GCN_Norm_M, stride, norm_layer)

        layers = []
        layers.append(block(self.inplanes, planes, M, GCN_Norm_M, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, M, GCN_Norm_M, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x, bs, t = modify_input(x)

        x = self.conv1(x)
        # x = bn(x, self.bn1, self.GCN_Norm_M, self.M)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x:[bs*t, c'*M, 1, 1]->[bs*t, c'*M, 1, 1]
        x = self.avgpool(x)

        # x:[bs*t, c'*M, 1, 1]->[bs, t, c', 1, 1]
        x = torch.max(x.view(bs * t, -1, self.M), dim=2)[0]
        # x:[bs*t, c', 1, 1]->[bs, t, c']
        x = x.reshape(bs, t, -1)

        # x:[bs, t, c']->[bs, t, num_classes]
        x = self.fc(x)
        # x:[bs, t, num_classes]->[bs, num_classes]
        x = torch.mean(x, dim=1)

        return x


def modify_input(input):
    # 读取视频数据
    if len(input.size()) == 6:
        # input.shape:[bs, num_segment, num_frame_in_segment, c, h, w]
        # 因为TSM是每一个segment里面抽取一帧
        assert input.shape[2] == 1, 'models.py/modify_input()'
        input = input.squeeze(dim=2)

        bs, t, c, h, w = input.size()
        input = input.reshape(bs * t, c, h, w)

        return input, bs, t

    elif len(input.size()) == 4:
        bs = input.size()[0]
        return input, bs, 1

    else:
        assert False, 'modify_input()'


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
