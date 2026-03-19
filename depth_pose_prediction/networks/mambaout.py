"""
MambaOut models for image classification.
Some implementations are modified from:
timm (https://github.com/rwightman/pytorch-image-models),
MetaFormer (https://github.com/sail-sg/metaformer),
InceptionNeXt (https://github.com/sail-sg/inceptionnext)
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

class StemLayer(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=96,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = norm_layer(out_channels // 2)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels // 2,
                               out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """
    def __init__(self, in_channels=96, out_channels=198, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, act_layer=nn.GELU, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class GDConv(nn.Module):
    def __init__(self, dim, dilation, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expension_ratio * dim)
        self.fc = nn.Linear(dim, hidden * 2)
        padding = (kernel_size // 2) * dilation
        self.dwconv = nn.Conv2d(
            hidden, hidden,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=hidden
        )

        self.conv1 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        f = self.fc(x) 

        fd = f.permute(0, 3, 1, 2)  # → [B, hidden, H, W]
        fd = self.dwconv(fd)
        fd = fd.permute(0, 2, 3, 1)  # → [B, H, W, hidden]

        fg = self.act(f)
        f_mul = fd * fg

        f_mul = f_mul.permute(0, 3, 1, 2)
        f_mul = self.conv1(f_mul)
        f_mul = f_mul.permute(0, 2, 3, 1)
        x = self.fc2(f_mul)
        
        x = self.drop_path(x)
        x = x + shortcut
        x = x.permute(0, 3, 2, 1)
        return x