import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
import torch.nn.functional as F
import numpy as np
import cv2
from mmcv.cnn import VGG
import warnings

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
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
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

    def forward(self, x):
        out = x
        out = self.channel_attention(out) * out
        out = self.spatial_attention(out) * out
        return x + out


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
        self.attions = CBAMOur(in_planes, ratio, kernel_size)

        self.contactconv1 = nn.Conv2d(2 * in_planes, in_planes, kernel_size=(1, 1), groups=in_planes, bias=False)
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



class SSDVGGmy(VGG, BaseModule):
    """VGG Backbone network for single-shot-detection.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_last_pool (bool): Whether to add a pooling layer at the last
            of the model
        ceil_mode (bool): When True, will use `ceil` instead of `floor`
            to compute the output shape.
        out_indices (Sequence[int]): Output from which stages.
        out_feature_indices (Sequence[int]): Output from which feature map.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
        input_size (int, optional): Deprecated argumment.
            Width and height of input, from {300, 512}.
        l2_norm_scale (float, optional) : Deprecated argumment.
            L2 normalization layer init scale.

    Example:
        >>> self = SSDVGG(input_size=300, depth=11)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 300, 300)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 19, 19)
        (1, 512, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 pretrained=None,
                 init_cfg=None,
                 input_size=None,
                 l2_norm_scale=None):
        # TODO: in_channels for mmcv.VGG
        super(SSDVGGmy, self).__init__(
            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices)

        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.out_feature_indices = out_feature_indices

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'

        if init_cfg is not None:
            self.init_cfg = init_cfg
        elif isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', val=1, layer='BatchNorm2d'),
                dict(type='Normal', std=0.01, layer='Linear'),
            ]
        else:
            raise TypeError('pretrained must be a str or None')

        if input_size is not None:
            warnings.warn('DeprecationWarning: input_size is deprecated')
        if l2_norm_scale is not None:
            warnings.warn('DeprecationWarning: l2_norm_scale in VGG is '
                          'deprecated, it has been moved to SSDNeck.')

    def init_weights(self, pretrained=None):
        super(VGG, self).init_weights()

    def forward(self, x):
        """Forward function."""
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)



@BACKBONES.register_module()
class fusionnet_CBAM_RGBD_SSD(BaseModule):
    def __init__(self,
                 init_cfg=None
                 ):
        super(fusionnet_CBAM_RGBD_SSD, self).__init__(init_cfg)

        self.rgbvgg = SSDVGGmy(input_size=512, depth=16, out_indices=(3, 4), out_feature_indices=(22, 34),
                        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://vgg16_caffe'))
        self.depthvgg = SSDVGGmy(input_size=512, depth=16, out_indices=(3, 4), out_feature_indices=(22, 34),
                          init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://vgg16_caffe'))
        # RGB Resnet 50
        # self.encodergb_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encodergb_bn1 = nn.BatchNorm2d(64)
        # self.encodergb_relu = nn.ReLU(inplace=True)
        # self.encodergb_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.encodergb_layer1 = _make_layer_bot(Bottleneck, 64, 3, stride=1, inplanes=64)
        # self.encodergb_layer2 = _make_layer_bot(Bottleneck, 128, 4, stride=2, inplanes=256)
        # self.encodergb_layer3 = _make_layer_bot(Bottleneck, 256, 6, stride=2, inplanes=512)
        # self.encodergb_layer4 = _make_layer_bot(Bottleneck, 512, 3, stride=2, inplanes=1024)

        # Depth Resnet 50
        # self.encodedepth_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encodedepth_bn1 = nn.BatchNorm2d(64)
        # self.encodedepth_relu = nn.ReLU(inplace=True)
        # self.encodedepth_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.encodedepth_layer1 = _make_layer_bot(Bottleneck, 64, 3, stride=1, inplanes=64)
        # self.encodedepth_layer2 = _make_layer_bot(Bottleneck, 128, 4, stride=2, inplanes=256)
        # self.encodedepth_layer3 = _make_layer_bot(Bottleneck, 256, 6, stride=2, inplanes=512)
        # self.encodedepth_layer4 = _make_layer_bot(Bottleneck, 512, 3, stride=2, inplanes=1024)

        # self.cbam_layer1 = CBAMCrossall(256)
        # self.cbam_layer2 = CBAMCrossall(512)
        # self.cbam_layer3 = CBAMCrossall(1024)
        # self.cbam_layer4 = CBAMCrossall(2048)
        #
        # self.cbam_layer1 = CBAMCrossOur(256, 16)
        self.cbam_layer1 = CBAMCrossOur(512, 16)
        self.cbam_layer2 = CBAMCrossOur(1024, 32)
        # self.cbam_layer4 = CBAMCrossOur(2048, 32)

        # self.skip_layer1 = nn.Identity()
        # self.skip_layer2 = nn.Identity()
        # self.skip_layer3 = nn.Identity()
        # self.skip_layer4 = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        depth = x[:, 3:, :, :]
        rgb = x[:, :3, :, :]

        [out_r1, out_r2] = self.rgbvgg(rgb)
        [out_d1, out_d2] = self.depthvgg(depth)

        # block 1
        skip1 = self.cbam_layer1(out_r1, out_d1)
        skip2 = self.cbam_layer2(out_r2, out_d2)

        return [skip1, skip2]

    def init_weights(self):

        # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/fasterrcnn2x.pth'
        # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/cascade.pth'
        # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/retinanet.pth'
        # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/resnet50.pth'
        # fasterrcnn = torch.load(fasterrcnnpath)

        fasterrcnnpath = 'D:/paper1/code/python/mmdetectionFCOS/checkpoints/ssd500.pth'
        fasterrcnn = torch.load(fasterrcnnpath)['state_dict']
        modeldata = self.state_dict()

        # Depth
        for k, v in fasterrcnn.items():
            if 'backbone' in k:
                k1 = k.replace('backbone.', 'rgbvgg.')
                if k1 in modeldata:
                    v1 = modeldata[k1]
                    if v.numel() == v1.numel() and len(v.shape) == len(v1.shape):
                        modeldata[k1].data.copy_(v.data)
                        print("copyparam" + k1)
                    else:
                        print("@@@@@@@@@@@" + k1)
                else:
                    print(k)

        # RGB
        for k, v in fasterrcnn.items():
            if 'backbone' in k:
                k1 = k.replace('backbone.', 'depthvgg.')
                if k1 in modeldata:
                    v1 = modeldata[k1]
                    if v.numel() == v1.numel() and len(v.shape) == len(v1.shape):
                        modeldata[k1].data.copy_(v.data)
                        print("copyparam" + k1)
                    else:
                        print("@@@@@@@@@@@" + k1)
                else:
                    print(k)


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
        super(fusionnet_CBAM_RGBD_SSD, self).train(mode)
        self._freeze_stages()
