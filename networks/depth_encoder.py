import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda
from einops import rearrange, reduce
from networks.basic_modules import get_norm, get_act, ConvNormAct, ConvNorm2d, MSPatchEmb

class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos



class TimeEmbeddingSine(nn.Module):
    """
    Same as below for temporal dimension
    """

    def __init__(self, max_len=2, d_model=512):
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        te = torch.zeros(max_len, 1, d_model)
        te[:, 0, 0::2] = torch.sin(position * div_term)
        te[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("te", te)

    def forward(self, ln):
        tim_t = self.te[:ln]
        return tim_t


class XCAf(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.k = 256
        self.linear_0 = nn.Conv1d(96, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, 96, 1, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # print(N)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # print(q.shape)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        B1, H1, S1, C1 = q.shape
        q1 = q.reshape(B1 * H1, S1, C1)
        k1 = k.reshape(B1 * H1, S1, C1)

        q1 = self.linear_0(q1)
        k1 = self.linear_0(k1)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        q1 = self.linear_1(q1)
        k1 = self.linear_1(k1)

        q = q1.reshape(B1, H1, S1, C1)
        k = k1.reshape(B1, H1, S1, C1)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class XCAt(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(256, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        # print(x.shape)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        # print(self.q(y).shape)
        q = self.q(y).reshape(B, N, 1, self.num_heads, C // self.num_heads)
        # print(N)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # print(k.shape, v.shape)
        q = q.permute(2, 0, 3, 1, 4)
        q = q[0]
        # print(q.shape)

        # print(q.shape)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.v_local = nn.Sequential(nn.Conv2d(self.num_heads, self.num_heads,
        #                                        kernel_size=3, stride=1, padding=1, groups=self.num_heads),
        #                              nn.BatchNorm2d(self.num_heads), )
        # self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        # self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        # self.k = 256
        # self.linear_0 = nn.Conv1d(96, self.k, 1, bias=False)
        # self.linear_1 = nn.Conv1d(self.k, 96, 1, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # print(N)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # print(q.shape)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        # print(v.shape)
        # v_local = self.v_local(v)

        # B1, H1, S1, C1 = q.shape
        # q1 = q.reshape(B1 * H1, S1, C1)
        # k1 = k.reshape(B1 * H1, S1, C1)
        #
        # q1 = self.linear_0(q1)
        # k1 = self.linear_0(k1)
        #
        # q1 = torch.nn.functional.normalize(q1, dim=-1)
        # k1 = torch.nn.functional.normalize(k1, dim=-1)
        #
        # q1 = self.linear_1(q1)
        # k1 = self.linear_1(k1)
        #
        # q = q1.reshape(B1, H1, S1, C1)
        # k = k1.reshape(B1, H1, S1, C1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # attn = self.talking_head1(attn)
        #
        # attn = attn.softmax(dim=-1)
        # attn = self.talking_head2(attn)
        # attn = self.attn_drop(attn)
        #
        # attn_n = (attn @ v) + v_local
        #
        # x = attn_n.permute(0, 3, 1, 2).reshape(B, N, C)

        # print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class XCAC(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.v_local = nn.Sequential(nn.Conv2d(self.num_heads, self.num_heads,
        #                                        kernel_size=3, stride=1, padding=1, groups=self.num_heads),
        #                              nn.BatchNorm2d(self.num_heads), )
        # self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        # self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        # self.k = 16
        # self.linear_0 = nn.Conv2d(8, self.k, 1, bias=False)
        # self.linear_1 = nn.Conv2d(self.k, 8, 1, bias=False)

    def forward(self, x, tem, pos_flag):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print(q.shape)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        # v_local = self.v_local(v)

        # q = self.linear_0(q)
        # k = self.linear_0(k)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # q = self.linear_1(q)
        # k = self.linear_1(k)
        if pos_flag:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
        else:
            # print(((q @ k.transpose(-2, -1)) * self.temperature).shape)
            attn = (q @ k.transpose(-2, -1)) * self.temperature + tem
        # print('attn2', attn.shape)
        # print('tem', tem.shape)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # attn = self.talking_head1(attn)
        #
        # attn = attn.softmax(dim=-1)
        # attn = self.talking_head2(attn)
        # attn = self.attn_drop(attn)
        #
        # attn_n = (attn @ v) + v_local
        #
        # x = attn_n.permute(0, 3, 1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """

    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)
        self.expan_ratio = 4
        # self.pool = nn.AdaptiveAvgPool2d()

        # self.norm = LayerNorm(dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(dim, 3 * dim)
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(3 * dim, dim)

        self.pwconv1 = nn.Conv2d(dim, dim*self.expan_ratio, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim*self.expan_ratio, dim, 1)
        # self.dwconv = nn.Conv2d(dim*self.expan_ratio, dim*self.expan_ratio, 3, 1, 1, bias=True, groups=dim)

        ## !!!!!!!!!
        # self.pwconv10 = nn.Conv2d(dim, dim, 1)
        # self.pwconv11 = nn.Conv2d(dim, self.expan_ratio * dim, 5, padding=2, groups=dim)
        # self.conv_spatial = nn.Conv2d(self.expan_ratio * dim, dim, 7, stride=1, padding=9,
        #                               groups=dim, dilation=3)
        # self.pwconv12 = nn.Conv2d(dim, dim, 1)
        # self.act = nn.GELU()


        # self.pwconv0 = nn.Conv2d(dim, dim, 1)
        # self.pwconv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # self.pwconv2 = nn.Conv2d(dim, dim, 1)
        # self.pwconv3 = nn.Conv2d(dim, dim, 1)
        # self.act = nn.GELU()

        # self.pwconv0 = ConvNorm2d(dim, dim, kernel_size=1, stride=1, dilation=0,
        #                               groups=1, norm_layer='bn_2d', act_layer='silu', inplace=True)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop = nn.Dropout(drop_path)

    def forward(self, x):
        input = x

        xt = self.ddwconv(x)
        x = self.bn1(xt)
        # x = self.act(x)
        # x = self.drop(x) + xt

        # xi = self.pwconv0(x)
        # ## attention
        # xi = self.pwconv1(xi)
        # xi = self.conv_spatial(xi)
        # xi = self.pwconv2(xi)
        #
        # xi = self.act(xi)
        # xi = self.pwconv3(xi)

        # xi = self.pwconv0(x)
        #
        # x = xi.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        # #


        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.pwconv2(x)

        ## !!!!!!
        # x_s = self.pwconv10(x)
        # x_s = self.pwconv11(x_s)
        # x_s = self.conv_spatial(x_s)
        # x_s = self.act(x_s)
        # x_s= x_s * input
        # x = self.pwconv12(x_s)


        # x_s = self.bn1(x_s)
        # x_c = x_s.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # x_c = self.pwconv1(x_s)
        # x_c = self.dwconv(x_c)
        # x_c = self.act(x_c)
        # x = self.pwconv2(x_c)

        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class LGFI(nn.Module):
    """
    Local-Global Features Interaction
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.expan_ration = 1
        self.pos_embd = None
        self.use_pos_emb = use_pos_emb
        # if use_pos_emb:
        self.pos_embd = PositionalEncodingFourier(dim=self.dim)

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCAC(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # self.norm = LayerNorm(self.dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)

        self.pwconv0 = ConvNormAct(self.dim, self.dim*self.expan_ration, kernel_size=3, stride=1, dilation=1,
                                   groups=1, norm_layer='bn_2d', act_layer='silu', inplace=True)
        # self.pwconv1 = ConvNormAct(self.dim, self.dim*self.expan_ration, kernel_size=1, stride=1, dilation=1,
        #                            groups=1, norm_layer='bn_2d', act_layer='silu', inplace=True)
        self.proj_drop = nn.Dropout(drop_path)
        # self.proj = ConvNormAct(self.dim*self.expan_ration, self.dim, kernel_size=1, norm_layer='none', act_layer='none', inplace=True)
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        # self.pwconv2 = ConvNormAct(self.dim, expan_ratio * self.dim, kernel_size=1, stride=1, dilation=1,
        # groups=1, norm_layer='bn_2d', act_layer='silu', inplace=True)


        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.temp_embd = TimeEmbeddingSine(48, 1)

        ## temproal
        self.seq_len = 1
        self.num_heads = num_heads
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * self.seq_len - 1, self.num_heads))  # 2*T-1, nH
        self.register_buffer("relative_position_index", self.get_position_index())  # T, T

    def get_position_index(self):
        coords = torch.arange(self.seq_len)  # T
        relative_coords = coords[:, None] - coords[None, :]  # T, T
        relative_coords += self.seq_len - 1  # shift to start from 0
        relative_position_index = relative_coords  # T, T
        return relative_position_index

    def forward(self, x):
        input_ = x

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.use_pos_emb:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            # pos_encoding = torch.mean(pos_encoding, dim=2, keepdim=True)
            # time_encoding = self.temp_embd(x.shape[2]).permute(2, 1, 0)
            # print(time_encoding.shape)

            # print(pos_encoding.shape)
            # x = x + pos_encoding + time_encoding
            x = x + pos_encoding

        T = self.seq_len
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)].reshape(T, T, -1)  # T, T, nH
        relative_position_bias = relative_position_bias[None].permute(3, 0, 1, 2).expand(self.num_heads, (C//self.num_heads) * (C//self.num_heads), T, T)
        relative_position_bias = F.pixel_shuffle(relative_position_bias, upscale_factor=int(C//self.num_heads)).permute(1, 0, 2, 3)

        # T = self.seq_len
        # relative_position_bias = self.relative_position_bias_table[
        #     self.relative_position_index.reshape(-1)].reshape(T, T, -1)  # T, T, nH
        # relative_position_bias = relative_position_bias[None].permute(3, 0, 1, 2).expand(self.num_heads,
        #                                                                                  (C // self.num_heads) * (
        #                                                                                              C // self.num_heads),
        #                                                                                  T, T)
        # # print(relative_position_bias.shape)
        # relative_position_bias = F.pixel_shuffle(relative_position_bias,
        #                                          upscale_factor=int(C // self.num_heads)).permute(1, 0, 2, 3)

        # print(relative_position_bias.shape)

        # relative_position_bias = relative_position_bias[None].permute(0, 3, 1, 2)
        # print(relative_position_bias.shape)

        # pos_position = self.pos_embd(1, C//self.num_heads, C//self.num_heads)
        # pos_position = torch.mean(pos_position, dim=1, keepdim=True)
        # # print(pos_position.shape)
        # relative_position_bias = torch.matmul(relative_position_bias, pos_position)
        # print(relative_position_bias.shape)

        # x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x + self.gamma_xca * self.xca(self.norm_xca(x), relative_position_bias, self.use_pos_emb)

        # x = x.reshape(B, H, W, C)

        x = x.reshape(B, C, H, W)
        # Inverted Bottleneck
        # x = self.norm(x)
        xi = self.pwconv0(x)
        # x = self.pwconv1(x)
        # print(xi.shape)
        # print(x.shape)
        # x = self.act(x)
        # x = self.pwconv2(x)
        # print(x.shape)
        x = x + (xi)
        # x = self.pwconv1(self.proj_drop(x))

        # x = self.proj_drop(x)
        # x = self.proj(x)

        x = x.reshape(B, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)

        return x


class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x


# class iRMB(nn.Module):
#
#     def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
#                  act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64, window_size=7,
#                  attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
#         super().__init__()
#         self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
#         dim_mid = int(dim_in * exp_ratio)
#         self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
#         self.attn_s = attn_s
#         if self.attn_s:
#             assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
#             self.dim_head = dim_head
#             self.window_size = window_size
#             self.num_head = dim_in // dim_head
#             self.scale = self.dim_head ** -0.5
#             self.attn_pre = attn_pre
#             self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
#                                   act_layer='none')
#             self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
#                                  norm_layer='none', act_layer=act_layer, inplace=inplace)
#             self.attn_drop = nn.Dropout(attn_drop)
#         else:
#             if v_proj:
#                 self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
#                                      act_layer=act_layer, inplace=inplace)
#             else:
#                 self.v = nn.Identity()
#         self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
#                                       groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
#         self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
#
#         self.proj_drop = nn.Dropout(drop)
#         self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
#         self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
#
#     def forward(self, x):
#         shortcut = x
#         x = self.norm(x)
#         B, C, H, W = x.shape
#         if self.attn_s:
#             # padding
#             if self.window_size <= 0:
#                 window_size_W, window_size_H = W, H
#             else:
#                 window_size_W, window_size_H = self.window_size, self.window_size
#             pad_l, pad_t = 0, 0
#             pad_r = (window_size_W - W % window_size_W) % window_size_W
#             pad_b = (window_size_H - H % window_size_H) % window_size_H
#             x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
#             n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
#             x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
#             # attention
#             b, c, h, w = x.shape
#             qk = self.qk(x)
#             qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
#                            dim_head=self.dim_head).contiguous()
#             q, k = qk[0], qk[1]
#             attn_spa = (q @ k.transpose(-2, -1)) * self.scale
#             attn_spa = attn_spa.softmax(dim=-1)
#             attn_spa = self.attn_drop(attn_spa)
#             if self.attn_pre:
#                 x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
#                 x_spa = attn_spa @ x
#                 x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
#                                   w=w).contiguous()
#                 x_spa = self.v(x_spa)
#             else:
#                 v = self.v(x)
#                 v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
#                 x_spa = attn_spa @ v
#                 x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
#                                   w=w).contiguous()
#             # unpadding
#             x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
#             if pad_r > 0 or pad_b > 0:
#                 x = x[:, :, :H, :W].contiguous()
#         else:
#             x = self.v(x)
#
#         x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
#
#         x = self.proj_drop(x)
#         x = self.proj(x)
#
#         x = (shortcut + self.drop_path(x)) if self.has_skip else x
#         return x


class LiteMono(nn.Module):
    """
    Lite-Mono
    """

    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()

        if model == 'lite-mono':
            # self.num_ch_enc = np.array([48, 80, 128])
            # self.depth = [4, 4, 10]
            # self.dims = [48, 80, 128]
            # if height == 192 and width == 640:
            #     self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            # elif height == 320 and width == 1024:
            #     self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]
            self.num_ch_enc = np.array([32, 64, 128])
            self.depth = [4, 4, 10]
            self.dims = [32, 64, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 7]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 128])
            self.depth = [4, 4, 7]
            self.dims = [32, 64, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-8m':
            self.num_ch_enc = np.array([64, 128, 224])
            self.depth = [4, 4, 10]
            self.dims = [64, 128, 224]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.stem2 = nn.Sequential(
            Conv(self.dims[0] + 3, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        self.downsample_layers.append(stem1)

        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i] * 2 + 3, self.dims[i + 1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(
                        DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value,
                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        x = (x - 0.45) / 0.225

        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))
        tmp_x.append(x)

        input_ = x

        for s in range(len(self.stages[0]) - 1):
            # x = self.stages[0][s](x)
            # if s == len(self.stages[0])-2:
            #     # x_ = torch.add(x, input_)
            #     x_ = self.stages[0][s](input_)
            #     x = x_ + x
            #     # print(11111)
            # else:
            x = self.stages[0][s](x)
        x = self.stages[0][-1](x)
        tmp_x.append(x)
        features.append(x)

        for i in range(1, 3):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)

            tmp_x = [x]

            input_ = x

            for s in range(len(self.stages[i]) - 1):
                # if s == len(self.stages[i]) - 2:
                #     # x_ = torch.add(x, input_)
                #     x_ = self.stages[i][s](input_)
                #     x = x_ + x
                # else:
                x = self.stages[i][s](x)
                # if s == len(self.stages[i]) - 2:
                #     x = torch.add(x, input_)
            x = self.stages[i][-1](x)
            tmp_x.append(x)

            features.append(x)

        return features

    def forward(self, x):
        x = self.forward_features(x)

        return x
