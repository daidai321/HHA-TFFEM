import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
import torch.nn.functional as F
import numpy as np
import cv2


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=4, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class jointSqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels, activation=nn.ReLU(inplace=True)):
        super(jointSqueezeAndExciteFusionAdd, self).__init__()

        self.semod = SqueezeAndExcitation(channels, activation=activation)
        self.alignconv = conv1x1(2 * channels, channels)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = torch.cat((rgb, depth), 1)
        out = self.semod(out)
        out = self.alignconv(out)
        return out


reducelist = {
    64: 32,
    256: 16,
    512: 32,
    1024: 64,
    2048: 128,
}


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in_rgb, channels_in_depth, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in_rgb, reduction=reducelist[channels_in_rgb],
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in_depth,
                                             activation=activation)
        self.alignconv = None
        if channels_in_rgb != channels_in_depth:
            self.alignconv = conv1x1(channels_in_depth, channels_in_rgb)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        if self.alignconv != None:
            out = rgb + self.alignconv(depth)
        else:
            out = rgb + depth
        return out, rgb, depth


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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


def _make_layer(block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                stride: int = 1, dilate: bool = False, inplanes: int = 64) -> nn.Sequential:
    norm_layer = nn.BatchNorm2d
    downsample = None
    if stride > 1:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, 1,
                        64, 1, norm_layer))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, groups=1,
                            base_width=64, dilation=1,
                            norm_layer=norm_layer))

    return nn.Sequential(*layers)


def _make_layer_bot(block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, inplanes: int = 64) -> nn.Sequential:
    norm_layer = nn.BatchNorm2d
    downsample = nn.Sequential(
        conv1x1(inplanes, planes * block.expansion, stride),
        norm_layer(planes * block.expansion),
    )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, 1,
                        64, 1, norm_layer))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, groups=1,
                            base_width=64, dilation=1,
                            norm_layer=norm_layer))

    return nn.Sequential(*layers)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                      relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    '''
    drived from:https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py#L25
    '''

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        out = avg_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes=in_planes, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x
        out = self.channel_attention(out) * out
        out = self.spatial_attention(out) * out
        return x + out


class CBAM_conv(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM_conv, self).__init__()
        self.channel_attention = ChannelAttention(in_planes=in_planes, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

        # self.baseconv = nn.Sequential(
        #     nn.Conv2d(in_planes, in_planes, (3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(in_planes),
        #     nn.ReLU(inplace=True)
        # )

        self.baseconv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, (1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #
        residual = x

        out = self.baseconv(x)
        out = self.channel_attention(out) * out
        out = self.spatial_attention(out) * out

        out += residual
        out = self.relu(out)

        return out


class CBAMCross(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMCross, self).__init__()
        self.attions = CBAM(in_planes, ratio, kernel_size)

        self.contactconv1 = nn.Conv2d(2 * in_planes, in_planes, kernel_size=1, groups=in_planes, bias=False)
        self.contactconv1_bn = nn.BatchNorm2d(in_planes)
        self.contactconv1_relu = nn.ReLU()

    def forward(self, rgb, depth):
        t, c, w, h = rgb.shape
        out = torch.cat((rgb, depth), 1)
        # out = torch.zeros((t, 2 * c, w, h))
        out[:, 0:2 * c - 1:2, :, :] = rgb
        out[:, 1:2 * c:2, :, :] = depth

        out = self.contactconv1_relu(self.contactconv1_bn(self.contactconv1(out)))
        out = self.attions(out)
        return out


class ChannelAttentionOur(nn.Module):
    '''
    drived from:https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py#L25
    '''

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionOur, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes // ratio, in_planes // ratio, (1, 1), bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False)

        self.sigmoid = nn.Sigmoid()

        self.contactconv1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=(3, 3), padding=(1, 1),
                                      groups=in_planes // ratio,
                                      bias=False)
        self.contactconv1_bn = nn.BatchNorm2d(in_planes // ratio)
        self.contactconv1_relu = nn.ReLU()

    def forward(self, x):
        x = self.contactconv1_relu(self.contactconv1_bn(self.contactconv1(x)))
        x = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(x)


class SpatialAttentionOur(nn.Module):
    def __init__(self, in_planes, kernel_size=7, ratio=16):
        super(SpatialAttentionOur, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size=(kernel_size, kernel_size), padding=(padding, padding), bias=False)
        self.sigmoid = nn.Sigmoid()

        # self.contactconv1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=(3, 3), padding=(1, 1),
        #                               groups=in_planes // ratio, bias=False)
        # self.contactconv1_bn = nn.BatchNorm2d(in_planes // ratio)
        # self.contactconv1_relu = nn.ReLU()

    def forward(self, x):
        # x = self.contactconv1_relu(self.contactconv1_bn(self.contactconv1(x)))

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMOur(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMOur, self).__init__()
        self.channel_attention = ChannelAttentionOur(in_planes=in_planes, ratio=ratio)
        self.spatial_attention = SpatialAttentionOur(in_planes, kernel_size, ratio)

        # self.RFB = BasicRFB_a(in_planes, in_planes)

    def forward(self, x):
        # x = self.RFB(x)
        out = x
        out = self.channel_attention(out) * out
        out = self.spatial_attention(out) * out
        return x + out


class Crossfusion_directly_multi(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(Crossfusion_directly_multi, self).__init__()

        self.down_contact = nn.Sequential(
            nn.Conv2d(2 * in_planes, in_planes, (1, 1), bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )

        self.donw_cross = nn.Sequential(
            nn.Conv2d(2 * in_planes, in_planes, (1, 1), groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )

        self.down_cross1 = nn.Sequential(
            nn.Conv2d(2 * in_planes, in_planes, (1, 1), groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )
        # self.attions = CBAMOur(in_planes, ratio, kernel_size)

    def forward(self, rgb, depth):
        x1 = torch.cat((rgb, depth), dim=1)
        x1 = self.down_contact(x1)

        t, c, w, h = rgb.shape
        x2 = torch.cat((rgb, depth), 1)
        x2[:, 0:2 * c - 1:2, :, :] = rgb
        x2[:, 1:2 * c:2, :, :] = depth
        x2 = self.donw_cross(x2)

        t, c, w, h = rgb.shape
        out = torch.cat((x1, x2), 1)
        out[:, 0:2 * c - 1:2, :, :] = x1
        out[:, 1:2 * c:2, :, :] = x2
        out = self.down_cross1(out)

        return out


class Crossfusion_directly(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(Crossfusion_directly, self).__init__()

        self.down_contact = nn.Sequential(
            nn.Conv2d(2 * in_planes, in_planes, (1, 1), bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )

        self.cross = nn.Sequential(
            nn.Conv2d(2 * in_planes, in_planes, (1, 1), groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )
        # self.attions = CBAMOur(in_planes, ratio, kernel_size)

    def forward(self, rgb, depth):
        x1 = torch.cat((rgb, depth), dim=1)
        x1 = self.down_contact(x1)

        t, c, w, h = rgb.shape
        x2 = torch.cat((rgb, depth), 1)
        x2[:, 0:2 * c - 1:2, :, :] = rgb
        x2[:, 1:2 * c:2, :, :] = depth
        x2 = self.cross(x2)

        out = torch.cat((x1, x2), dim=1)

        # out = self.attions(out)
        return out


class Crossfusion_gc(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(Crossfusion_gc, self).__init__()

        self.down_rgb = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, (1, 1), bias=False),
            nn.BatchNorm2d(in_planes // 2),
            nn.ReLU(inplace=True)
        )
        self.down_depth = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, (1, 1), bias=False),
            nn.BatchNorm2d(in_planes // 2),
            nn.ReLU(inplace=True)
        )

        self.cont_conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(in_planes // 2),
            nn.ReLU(inplace=True)
        )

        self.cross_conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, (3, 3), padding=(1, 1), groups=in_planes // 2, bias=False),
            nn.BatchNorm2d(in_planes // 2),
            nn.ReLU(inplace=True)
        )

        self.attions = CBAM_conv(in_planes, ratio, kernel_size)

    def forward(self, rgb, depth):
        rgbd = self.down_rgb(rgb)
        depthd = self.down_depth(depth)
        x1 = torch.cat([rgbd, depthd], dim=1)
        x1 = self.cont_conv(x1)

        t, c, w, h = rgbd.shape
        x2 = torch.cat((rgbd, depthd), 1)
        x2[:, 0:2 * c - 1:2, :, :] = rgbd
        x2[:, 1:2 * c:2, :, :] = depthd
        x2 = self.cross_conv(x2)

        out = torch.cat((x1, x2), dim=1)

        # out = torch.cat((rgb, depth), dim=1)

        out = self.attions(out)

        return out


# def savetensorx(x, savepath):
#     # x2s = torch.max(x, 1)
#     x2s = torch.mean(x, dim=1, keepdim=True)
#     # x2s, _ = torch.max(x, dim=1, keepdim=True)
#     x2s1 = x2s[0, 0, :, :]
#     x2s1 = x2s1.cpu().numpy()
#     savep = (x2s1 - np.min(x2s1)) / (np.max(x2s1) - np.min(x2s1))
#     savep = (savep * 255).astype(np.uint8)
#     cv2.imwrite(savepath, savep)


class CBAMCrossOur(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMCrossOur, self).__init__()
        # self.attions = CBAMOur(in_planes, ratio, kernel_size)
        self.attions = CBAM_conv(in_planes, ratio, kernel_size)

        # self.contactconv1 = nn.Conv2d(2 * in_planes, in_planes, kernel_size=(1, 1), groups=in_planes, bias=False)
        # self.contactconv1_bn = nn.BatchNorm2d(in_planes)
        # self.contactconv1_relu = nn.ReLU()

        self.contactconv2 = nn.Conv2d(2 * in_planes, in_planes, kernel_size=(1, 1), bias=False)
        self.contactconv2_bn = nn.BatchNorm2d(in_planes)
        self.contactconv2_relu = nn.ReLU()

    def forward(self, rgb, depth):
        # channel cross fusion
        # t, c, w, h = rgb.shape
        # out = torch.cat((rgb, depth), 1)
        # out[:, 0:2 * c - 1:2, :, :] = rgb
        # out[:, 1:2 * c:2, :, :] = depth
        # out = self.contactconv1_relu(self.contactconv1_bn(self.contactconv1(out)))
        # out = self.contactconv2_relu(self.contactconv2_bn(self.contactconv2(out)))

        # add fusion
        # out = rgb + depth

        # contact fusion
        out = torch.cat((rgb, depth), 1)
        out = self.contactconv2_relu(self.contactconv2_bn(self.contactconv2(out)))

        # out = self.attions(out)
        return out


@BACKBONES.register_module()
class fusionnet_CBAM_RGBD(BaseModule):
    def __init__(self,
                 init_cfg=None
                 ):
        super(fusionnet_CBAM_RGBD, self).__init__(init_cfg)

        # RGB Resnet 50
        self.encodergb_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encodergb_bn1 = nn.BatchNorm2d(64)
        self.encodergb_relu = nn.ReLU(inplace=True)
        self.encodergb_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encodergb_layer1 = _make_layer_bot(Bottleneck, 64, 3, stride=1, inplanes=64)
        self.encodergb_layer2 = _make_layer_bot(Bottleneck, 128, 4, stride=2, inplanes=256)
        self.encodergb_layer3 = _make_layer_bot(Bottleneck, 256, 6, stride=2, inplanes=512)
        self.encodergb_layer4 = _make_layer_bot(Bottleneck, 512, 3, stride=2, inplanes=1024)

        # Depth Resnet 50
        self.encodedepth_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encodedepth_bn1 = nn.BatchNorm2d(64)
        self.encodedepth_relu = nn.ReLU(inplace=True)
        self.encodedepth_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encodedepth_layer1 = _make_layer_bot(Bottleneck, 64, 3, stride=1, inplanes=64)
        self.encodedepth_layer2 = _make_layer_bot(Bottleneck, 128, 4, stride=2, inplanes=256)
        self.encodedepth_layer3 = _make_layer_bot(Bottleneck, 256, 6, stride=2, inplanes=512)
        self.encodedepth_layer4 = _make_layer_bot(Bottleneck, 512, 3, stride=2, inplanes=1024)

        # self.cbam_layer1 = CBAMCrossall(256)
        # self.cbam_layer2 = CBAMCrossall(512)
        # self.cbam_layer3 = CBAMCrossall(1024)
        # self.cbam_layer4 = CBAMCrossall(2048)

        # self.cbam_layer1 = CBAMCrossOur(256, 16)
        # self.cbam_layer2 = CBAMCrossOur(512, 16)
        # self.cbam_layer3 = CBAMCrossOur(1024, 16)
        # self.cbam_layer4 = CBAMCrossOur(2048, 16)

        self.cbam_layer1 = Crossfusion_gc(256, 16)
        self.cbam_layer2 = Crossfusion_gc(512, 16)
        self.cbam_layer3 = Crossfusion_gc(1024, 16)
        self.cbam_layer4 = Crossfusion_gc(2048, 16)

        self.skip_layer1 = nn.Identity()
        self.skip_layer2 = nn.Identity()
        self.skip_layer3 = nn.Identity()
        self.skip_layer4 = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        depth = x[:, 3:, :, :]
        rgb = x[:, :3, :, :]

        rgb = self.encodergb_relu(self.encodergb_bn1(self.encodergb_conv1(rgb)))
        depth = self.encodedepth_relu(self.encodedepth_bn1(self.encodedepth_conv1(depth)))
        rgb = self.encodergb_maxpool(rgb)
        depth = self.encodedepth_maxpool(depth)

        # block 1
        rgb = self.encodergb_layer1(rgb)
        depth = self.encodedepth_layer1(depth)

        skip1 = self.cbam_layer1(rgb, depth)
        skip1 = self.skip_layer1(skip1)

        # block 2
        rgb = self.encodergb_layer2(rgb)
        depth = self.encodedepth_layer2(depth)

        skip2 = self.cbam_layer2(rgb, depth)
        skip2 = self.skip_layer2(skip2)

        # block 3
        rgb = self.encodergb_layer3(rgb)
        depth = self.encodedepth_layer3(depth)
        skip3 = self.cbam_layer3(rgb, depth)
        skip3 = self.skip_layer3(skip3)

        # block 4
        rgb = self.encodergb_layer4(rgb)
        depth = self.encodedepth_layer4(depth)
        skip4 = self.cbam_layer4(rgb, depth)
        skip4 = self.skip_layer4(skip4)

        return [skip1, skip2, skip3, skip4]

    def init_weights(self):
        pass
        # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/fasterrcnn2x.pth'
        # otherrcnnpath = 'D:/paper_segmentation/code/mmdetection/tools/work_dirs/faster_rcnn_r50_fpn_1x_coco_paper/871_515.pth'
        # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/maskrcnn.pth'
        # # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/retinanet.pth'
        # # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/resnet50.pth'
        # # fasterrcnn = torch.load(fasterrcnnpath)
        # # fasterrcnnpath = 'D:/paper_s//egmentation/code/mmdetection/checkpoints/fcos.pth'
        #
        # fasterrcnn = torch.load(fasterrcnnpath)['state_dict']
        # modeldata = self.state_dict()
        #
        # otherrcnn = torch.load(otherrcnnpath)['state_dict']
        #
        # # Other
        # for k, v in otherrcnn.items():
        #     k1 = k
        #     if k1 in modeldata:
        #         v1 = modeldata[k1]
        #         if v.numel() == v1.numel() and len(v.shape) == len(v1.shape):
        #             modeldata[k1].data.copy_(v.data)
        #         else:
        #             print("kkkkkkkkkkkkkkkkk" + k1)
        #     else:
        #         print(k1)
        #
        # # Depth
        # for k, v in fasterrcnn.items():
        #     if 'backbone' in k:
        #         k1 = k.replace('backbone.', 'encodedepth_')
        #         if k1 in modeldata:
        #             v1 = modeldata[k1]
        #             if v.numel() == v1.numel() and len(v.shape) == len(v1.shape):
        #                 modeldata[k1].data.copy_(v.data)
        #                 print("copyparam" + k1)
        #             else:
        #                 print("@@@@@@@@@@@" + k1)
        #         else:
        #             print(k)
        #
        # # RGB
        # for k, v in fasterrcnn.items():
        #     if 'backbone' in k:
        #         k1 = k.replace('backbone.', 'encodergb_')
        #         if k1 in modeldata:
        #             v1 = modeldata[k1]
        #             if v.numel() == v1.numel() and len(v.shape) == len(v1.shape):
        #                 modeldata[k1].data.copy_(v.data)
        #                 print("copyparam" + k1)
        #             else:
        #                 print("@@@@@@@@@@@" + k1)
        #         else:
        #             print(k)

    def _freeze_stages(self):
        frozenlist = []
        # frozenlistrgb = [self.encodergb_conv1, self.encodergb_bn1, self.encodergb_relu, self.encodergb_maxpool,
        #                  self.encodergb_layer1, self.encodergb_layer2, self.encodergb_layer3, self.encodergb_layer4]
        #
        # frozenlistdepth = [self.encodedepth_conv1, self.encodedepth_bn1, self.encodedepth_relu,
        #                    self.encodedepth_maxpool,
        #                    self.encodedepth_layer1, self.encodedepth_layer2, self.encodedepth_layer3,
        #                    self.encodedepth_layer4]
        #
        # fronzenselayer = [self.se_layer0, self.se_layer1, self.se_layer2, self.se_layer3, self.se_layer4]
        #
        # for m in frozenlistrgb:
        #     m.eval()
        #     for param in m.parameters():
        #         param.requires_grad = False
        #
        # for m in frozenlistdepth:
        #     m.eval()
        #     for param in m.parameters():
        #         param.requires_grad = False
        #
        # for m in fronzenselayer:
        #     m.eval()
        #     for param in m.parameters():
        #         param.requires_grad = False

    def train(self, mode=True):
        super(fusionnet_CBAM_RGBD, self).train(mode)
        self._freeze_stages()
