# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from torch import nn


from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone
from .mobilenet_v2 import InvertedResidual
from .vision_transformer import TransformerEncoderLayer

import torch
import torch.nn as nn

class EfficientChannelAttention(nn.Module):
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute the average pooling across channels
        x_avg = self.avg_pool(x)

        # Calculate channel attention weights using convolution
        x_weights = self.conv(x_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Apply the sigmoid function to the attention weights
        attention_weights = self.sigmoid(x_weights)

        # Multiply the original input by the attention weights to emphasize important channels
        x_out = x * attention_weights

        return x_out

class ConvConcatModule(nn.Module):
    def __init__(self, in_channels, scales):
        super(ConvConcatModule, self).__init__()
        self.in_channels = in_channels
        self.scales = scales

        self.conv_channels = max(
            int(math.ceil(in_channels / scales)),
            int(math.floor(in_channels // scales))
        )

        self.num_convs = scales if scales == 1 else scales - 1

        self.conv_modules = nn.ModuleList([
            nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(self.num_convs)
        ])

    def forward(self, x):
        spx = torch.split(x, self.conv_channels, dim=1)
        out = None
        for i in range(self.num_convs):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.conv_modules[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        x = torch.cat((out, spx[self.num_convs]), 1)
        EfficientChannelAttention(x)
        return x


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
class ConvConcatModule(nn.Module):
    def __init__(self, in_channels, scales):
        super(ConvConcatModule, self).__init__()
        self.in_channels = in_channels
        self.scales = scales

        self.conv_channels = max(
            int(math.ceil(in_channels / scales)),
            int(math.floor(in_channels // scales))
        )

        self.num_convs = scales if scales == 1 else scales - 1

        self.conv_modules = nn.ModuleList([
            nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(self.num_convs)
        ])

    def forward(self, x):
        spx = torch.split(x, self.conv_channels, dim=1)
        out = None
        for i in range(self.num_convs):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.conv_modules[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        x = torch.cat((out, spx[self.num_convs]), 1)
        return x


class MobileVitBlock(nn.Module):
    """MobileViT block.

    According to the paper, the MobileViT block has a local representation.
    a transformer-as-convolution layer which consists of a global
    representation with unfolding and folding, and a final fusion layer.

    Args:
        in_channels (int): Number of input image channels.
        in_channels （int）： 输入图像通道数。
        transformer_dim (int): Number of transformer channels.
        transformer_dim (int)： 转换器通道数。
        ffn_dim (int): Number of ffn channels in transformer block.
        ffn_dim (int)： 转换器块中的 ffn 通道数。
        out_channels (int): Number of channels in output.
        out_channels (int)： 输出中的通道数。
        conv_ksize (int): Conv kernel size in local representation
            and fusion. Defaults to 3.
            conv_ksize (int)： 本地表示和融合中的 Conv 内核大小。
            和融合。默认为 3。
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
            conv_cfg （dict，可选）： 卷积层的配置指令。
            默认为 "无"，即使用 conv2d。
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
            norm_cfg（dict，可选）： 配置归一化层的 dict。
            默认为 dict(type='BN')。
        act_cfg (dict, optional): Config dict for activation layer.
            Defaults to dict(type='Swish').
            act_cfg （dict，可选）：激活层的配置 dict： 激活层的配置指令。
            默认为 dict(type='Swish')。
        num_transformer_blocks (int): Number of transformer blocks in
            a MobileViT block. Defaults to 2.
            num_transformer_blocks（整数）： 一个
            块中变压器块的数量。默认为 2。
        patch_size (int): Patch size for unfolding and folding.
             Defaults to 2.
              patch_size（int）： 用于展开和折叠的补丁大小。
             默认为 2。
        num_heads (int): Number of heads in global representation.
             Defaults to 4.
             num_heads (int)： 全局表示中的头部数量。
             默认为 4。
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
            drop_rate (float)： 元素在前馈层
            的概率。默认为 0。
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
            attn_drop_rate（浮点数）： 注意输出权重的丢弃率。
            默认为 0。
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        drop_path_rate（浮点数）： 随机深度率。默认为 0。
        no_fusion (bool): Whether to remove the fusion layer.
            Defaults to False.
            no_fusion（bool）： 是否移除融合层： 是否移除融合层。
            默认为 False。
        transformer_norm_cfg (dict, optional): Config dict for normalization
            layer in transformer. Defaults to dict(type='LN').
            参数：
        transformer_norm_cfg（dict，可选）： 转换器中规范化
            层的配置。默认为 dict(type='LN')。



    """

    def __init__(
            self,
            in_channels: int,
            transformer_dim: int,
            ffn_dim: int,
            out_channels: int,
            conv_ksize: int = 3,
            fusion_conv_ksize: int = 3,
            conv_cfg: Optional[dict] = None,
            norm_cfg: Optional[dict] = dict(type='BN'),
            act_cfg: Optional[dict] = dict(type='Swish'),
            num_transformer_blocks: int = 2,
            patch_size: int = 2,
            num_heads: int = 4,
            scales: int = 4,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            no_fusion: bool = False,
            transformer_norm_cfg: Callable = dict(type='LN'),
    ):
        super(MobileVitBlock, self).__init__()

        self.local_rep = nn.Sequential(
            ConvConcatModule(in_channels, scales),
            # print(scales),
            ConvModule(
                in_channels=in_channels,
                out_channels=transformer_dim,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None),
        )

        global_rep = [
            TransformerEncoderLayer(
                embed_dims=transformer_dim,
                num_heads=num_heads,
                feedforward_channels=ffn_dim,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                qkv_bias=True,
                act_cfg=dict(type='Swish'),
                norm_cfg=transformer_norm_cfg)
            for _ in range(num_transformer_blocks)
        ]
        global_rep.append(
            build_norm_layer(transformer_norm_cfg, transformer_dim)[1])
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = ConvModule(
            in_channels=transformer_dim,
            out_channels=out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = ConvModule(
                in_channels=transformer_dim + out_channels,
                out_channels=out_channels,
                kernel_size=fusion_conv_ksize,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        self.patch_size = (patch_size, patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Local representation
        x = self.local_rep(x)
        loca_shortcut = x
        # Unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(
            W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w # noqa
        num_patches = num_patch_h * num_patch_w  # N
        interpolate = False
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function. # noqa
            x = F.interpolate(
                x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            interpolate = True

        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w,
                      patch_w).transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w # noqa
        x = x.reshape(B, C, num_patches,
                      self.patch_area).transpose(1, 3).reshape(
                          B * self.patch_area, num_patches, -1)

        # Global representations BP, N, C

        x = self.global_rep(x)

        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w,
                                      patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W] # noqa
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h,
                                      num_patch_w * patch_w)
        if interpolate:
            x = F.interpolate(
                x, size=(H, W), mode='bilinear', align_corners=False)

        x = self.conv_proj(x)
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((loca_shortcut, x), dim=1))
        x = x + shortcut
        return x


@MODELS.register_module()
class MobileViT6(BaseBackbone):
    """MobileViT backbone.

    A PyTorch implementation of : `MobileViT: Light-weight, General-purpose,
    and Mobile-friendly Vision Transformer <https://arxiv.org/pdf/2110.02178.pdf>`_

    Modified from the `official repo
    <https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilevit.py>`_.

    Args:
        arch (str | List[list]): Architecture of MobileViT.

            - If a string, choose from "small", "x_small" and "xx_small".

            - If a list, every item should be also a list, and the first item
              of the sub-list can be chosen from "moblienetv2" and "mobilevit",
              which indicates the type of this layer sequence. If "mobilenetv2",
              the other items are the arguments of :attr:`~MobileViT.make_mobilenetv2_layer`
              (except ``in_channels``) and if "mobilevit", the other items are
              the arguments of :attr:`~MobileViT.make_mobilevit_layer`
              (except ``in_channels``).

            Defaults to "small".
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_channels (int): Channels of stem layer.  Defaults to 16.
        last_exp_factor (int): Channels expand factor of last layer.
            Defaults to 4.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (4, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer.
            Defaults to dict(type='Swish').
        init_cfg (dict, optional): Initialization config dict.
    """  # noqa

    # Parameters to build layers. The first param is the type of layer.
    # For `mobilenetv2` layer, the rest params from left to right are:
    #     out channels, stride, num of blocks, expand_ratio.
    # For `mobilevit` layer, the rest params from left to right are:
    #     out channels, stride, transformer_channels, ffn channels,
    # num of transformer blocks, expand_ratio.
    # 构建图层的参数。第一个参数是层的类型。
    # 对于 `mobilenetv2` 图层，其余参数从左到右依次为
    # 输出通道、跨度、块数、展开比。
    # 对于 `mobilevit` 层，其余参数从左到右依次为
    # 输出通道、跨接、变压器_通道、输入通道、输出通道
    # 变压器块数、expand_ratio。
    arch_settings = {
        'small': [
            ['mobilenetv2', 32, 1, 1, 4],
            ['mobilenetv2', 64, 2, 3, 4],
            ['mobilevit', 96, 2, 144, 288, 3, 2, 4, ],
            ['mobilevit', 128, 2, 192, 384, 4, 4, 4, ],
            ['mobilevit', 160, 2, 240, 480, 5, 3, 4, ],
        ],
        'x_small': [
            ['mobilenetv2', 32, 1, 1, 4],
            ['mobilenetv2', 48, 2, 3, 4],
            ['mobilevit', 64, 2, 96, 192, 3, 2, 4],
            ['mobilevit', 80, 2, 120, 240, 4, 4, 4],
            ['mobilevit', 96, 2, 144, 288, 5, 3, 4],
        ],
        'xx_small': [
            ['mobilenetv2', 16, 1, 1, 2],
            ['mobilenetv2', 24, 2, 3, 2],
            ['mobilevit', 48, 2, 64, 128, 3, 2, 2],
            ['mobilevit', 64, 2, 80, 160, 4, 4, 2],
            ['mobilevit', 80, 2, 96, 192, 5, 3, 2],
        ]
    }

    def __init__(self,
                 arch='small',
                 in_channels=3,
                 stem_channels=16,
                 last_exp_factor=4,
                 out_indices=(4,),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Swish'),
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(MobileViT6, self).__init__(init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a list.'
            arch = self.arch_settings[arch]

        self.arch = arch
        self.num_stages = len(arch)

        # check out indices and frozen stages
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        if frozen_stages not in range(-1, self.num_stages):
            raise ValueError('frozen_stages must be in range(-1, '
                             f'{self.num_stages}). '
                             f'But received {frozen_stages}')
        self.frozen_stages = frozen_stages

        _make_layer_func = {
            'mobilenetv2': self.make_mobilenetv2_layer,
            'mobilevit': self.make_mobilevit_layer,
        }

        self.stem = nn.Sequential(ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg),
            CoordAtt(stem_channels, stem_channels))

        in_channels = stem_channels
        layers = []
        for i, layer_settings in enumerate(arch):
            layer_type, settings = layer_settings[0], layer_settings[1:]
            layer, out_channels = _make_layer_func[layer_type](in_channels,
                                                               *settings)
            layers.append(layer)
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

        self.conv_1x1_exp = ConvModule(
            in_channels=in_channels,
            out_channels=last_exp_factor * in_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    @staticmethod
    def make_mobilevit_layer(in_channels,
                             out_channels,
                             stride,
                             transformer_dim,
                             ffn_dim,
                             scales,
                             num_transformer_blocks,
                             expand_ratio=4,
                             ):
        """Build mobilevit layer, which consists of one InvertedResidual and
        one MobileVitBlock.

        Args:
            scales:
            in_channels (int): The input channels.
            out_channels (int): The output channels.
            stride (int): The stride of the first 3x3 convolution in the
                ``InvertedResidual`` layers.
            transformer_dim (int): The channels of the transformer layers.
            ffn_dim (int): The mid-channels of the feedforward network in
                transformer layers.
            num_transformer_blocks (int): The number of transformer blocks.
            expand_ratio (int): adjusts number of channels of the hidden layer
                in ``InvertedResidual`` by this amount. Defaults to 4.
        """
        layer = []
        layer.append(
            InvertedResidual(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                act_cfg=dict(type='Swish'),
            ))
        layer.append(
            MobileVitBlock(
                in_channels=out_channels,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                scales=scales,
                out_channels=out_channels,
                num_transformer_blocks=num_transformer_blocks,
            ))
        return nn.Sequential(*layer), out_channels

    @staticmethod
    def make_mobilenetv2_layer(in_channels,
                               out_channels,
                               stride,
                               num_blocks,
                               expand_ratio=4):
        """Build mobilenetv2 layer, which consists of several InvertedResidual
        layers.

        Args:
            in_channels (int): The input channels.
            out_channels (int): The output channels.
            stride (int): The stride of the first 3x3 convolution in the
                ``InvertedResidual`` layers.
            num_blocks (int): The number of ``InvertedResidual`` blocks.
            expand_ratio (int): adjusts number of channels of the hidden layer
                in ``InvertedResidual`` by this amount. Defaults to 4.
        """
        layer = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1

            layer.append(
                InvertedResidual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    act_cfg=dict(type='Swish'),
                ))
            in_channels = out_channels
        return nn.Sequential(*layer), out_channels

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages):
            layer = self.layers[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileViT6, self).train(mode)
        self._freeze_stages()

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                x = self.conv_1x1_exp(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
