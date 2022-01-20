import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
import torch.nn.functional as F


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
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = x
        # out = self.channel_attention(out) * out
        out = self.spatial_attention(out) * out
        return x + out


@BACKBONES.register_module()
class FusionNetCBAMSIGNLE(BaseModule):
    def __init__(self,
                 init_cfg=None
                 ):
        super(FusionNetCBAMSIGNLE, self).__init__(init_cfg)

        # Depth Resnet 50
        self.encodedepth_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encodedepth_bn1 = nn.BatchNorm2d(64)
        self.encodedepth_relu = nn.ReLU(inplace=True)
        self.encodedepth_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encodedepth_layer1 = _make_layer_bot(Bottleneck, 64, 3, stride=1, inplanes=64)
        self.encodedepth_layer2 = _make_layer_bot(Bottleneck, 128, 4, stride=2, inplanes=256)
        self.encodedepth_layer3 = _make_layer_bot(Bottleneck, 256, 6, stride=2, inplanes=512)
        self.encodedepth_layer4 = _make_layer_bot(Bottleneck, 512, 3, stride=2, inplanes=1024)

        # self.cbam_layer0 = CBAM(64, 64)
        # self.cbam_layer1 = CBAM(256, 256)
        # self.cbam_layer2 = CBAM(512, 512)
        # self.cbam_layer3 = CBAM(1024, 1024)
        # self.cbam_layer4 = CBAM(2048, 2048)

        # self.cbam_layer0 = CBAM(64)
        self.cbam_layer1 = CBAM(256)
        self.cbam_layer2 = CBAM(512)
        self.cbam_layer3 = CBAM(1024)
        self.cbam_layer4 = CBAM(2048)

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

        depth = self.encodedepth_relu(self.encodedepth_bn1(self.encodedepth_conv1(depth)))
        # depth = self.cbam_layer0(depth)
        depth = self.encodedepth_maxpool(depth)

        # block 1
        depth = self.encodedepth_layer1(depth)
        skip1 = self.cbam_layer1(depth)
        skip1 = self.skip_layer1(skip1)

        # block 2
        depth = self.encodedepth_layer2(depth)
        skip2 = self.cbam_layer2(depth)
        skip2 = self.skip_layer2(skip2)

        # block 3
        depth = self.encodedepth_layer3(depth)
        skip3 = self.cbam_layer3(depth)
        skip3 = self.skip_layer3(skip3)

        # block 4
        depth = self.encodedepth_layer4(depth)
        skip4 = self.cbam_layer4(depth)
        skip4 = self.skip_layer4(skip4)

        return [skip1, skip2, skip3, skip4]

    def init_weights(self):

        # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/fasterrcnn2x.pth'
        # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/yolof.pth'
        # fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/retinanet.pth'

        fasterrcnnpath = 'D:/paper_segmentation/code/mmdetection/checkpoints/resnet50.pth'
        fasterrcnn = torch.load(fasterrcnnpath)
        # fasterrcnn = torch.load(fasterrcnnpath)['state_dict']
        modeldata = self.state_dict()

        # # Depth
        # for k, v in resnet18.items():
        #     k1 = 'encodedepth_' + k
        #     if k1 in modeldata:
        #         v1 = modeldata[k1]
        #         if v.numel() == v1.numel() and len(v.shape) == len(v1.shape):
        #             modeldata[k1].data.copy_(v.data)
        #         else:
        #             print("@@@@@@@@@@@" + k1)
        #     else:
        #         print(k1)

        # Depth
        for k, v in fasterrcnn.items():
            if 'backbone' in k:
                k1 = k.replace('backbone.', 'encodedepth_')
                if k1 in modeldata:
                    v1 = modeldata[k1]
                    if v.numel() == v1.numel() and len(v.shape) == len(v1.shape):
                        modeldata[k1].data.copy_(v.data)
                    else:
                        print("@@@@@@@@@@@" + k1)
                else:
                    print(k)

        # RGB
        for k, v in fasterrcnn.items():
            if 'backbone' in k:
                k1 = k.replace('backbone.', 'encodergb_')
                if k1 in modeldata:
                    v1 = modeldata[k1]
                    if v.numel() == v1.numel() and len(v.shape) == len(v1.shape):
                        modeldata[k1].data.copy_(v.data)
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
        super(FusionNetCBAMSIGNLE, self).train(mode)
        self._freeze_stages()
