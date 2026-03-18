import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import networks.depth_encoder
from timm.models.layers import DropPath
import math
from networks.basic_modules import ConvNormAct1d, ConvNorm1d


# The inertial encoder for raw imu data
# class Inertial_encoder(nn.Module):
#     def __init__(self, opt):
#         super(Inertial_encoder, self).__init__()
#
#         self.encoder_conv = nn.Sequential(
#             nn.Conv1d(6, 64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(opt.imu_dropout),
#             nn.Conv1d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(opt.imu_dropout),
#             nn.Conv1d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(opt.imu_dropout))
#         self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)
#
#     def forward(self, x):
#         # x: (N, seq_len, 11, 6)
#         batch_size = x.shape[0]
#         # seq_len = x.shape[1]
#         # x = x.view(batch_size, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
#         # x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
#         x = self.encoder_conv(x.permute(0, 2, 1).float())
#         out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
#         return out.view(batch_size, 256)

# class PositionalEncodingFourier1d(nn.Module):
#     """
#     Positional encoding relying on a fourier kernel matching the one used in the
#     "Attention is all of Need" paper. The implementation builds on DeTR code
#     https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
#     """

#     def __init__(self, hidden_dim=32, dim=768, temperature=10000):
#         super().__init__()
#         self.token_projection = nn.Conv1d(hidden_dim, dim, kernel_size=1)
#         self.scale = 2 * math.pi
#         self.temperature = temperature
#         self.hidden_dim = hidden_dim
#         self.dim = dim

#     def forward(self, B, L):
#         mask = torch.zeros(B, L).bool().to(self.token_projection.weight.device)
#         not_mask = ~mask
#         y_embed = not_mask.cumsum(1, dtype=torch.float32)
#         eps = 1e-6
#         y_embed = y_embed / (y_embed[:, -1:].contiguous() + eps) * self.scale

#         dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

#         pos_y = y_embed[:, :, None].contiguous() / dim_t
#         print('p', pos_y.shape)
       
#         pos_y = torch.stack((pos_y[:, :, 0::2].contiguous().sin(),
#                              pos_y[:, :, 1::2].contiguous().cos()), dim=3).flatten(2)
#         pos = pos_y.permute(0, 2, 1).contiguous()
#         pos = self.token_projection(pos)
#         print(pos.shape)
#         return pos

class PositionalEncodingFourier1d(nn.Module):

    def __init__(self, d_hid=11, dim=256):
        super(PositionalEncodingFourier1d, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(dim, d_hid))

    def _get_sinusoid_encoding_table(self, dim, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(dim)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()

class TimeEmbeddingSine(nn.Module):
    """
    Same as below for temporal dimension
    """

    def __init__(self, max_len=200, d_model=512):
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
        pos_t = self.te[:ln]
        return pos_t

class TimeEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=200, d_model=512):
        super().__init__()
        self.time_embed = nn.Embedding(num_pos_feats, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.time_embed.weight)

    def forward(self, ln):
        return self.time_embed.weight[:ln].unsqueeze(1)

# class XCA(nn.Module):
#     """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
#      sum. The weights are obtained from the (softmax normalized) Cross-covariance
#     matrix (Q^T K \\in d_h \\times d_h)
#     """

#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
#         qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         q = q.transpose(-2, -1).contiguous()
#         k = k.transpose(-2, -1).contiguous()
#         v = v.transpose(-2, -1).contiguous()

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).permute(0, 3, 1, 2).contiguous().reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'temperature'}

class CDilated1d(nn.Module):
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
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output

class DilatedConv1d(nn.Module):
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

        self.ddwconv = CDilated1d(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm1d(dim)
        self.expan_ratio = 4

        # self.pwconv1 = nn.Linear(dim, 3 * dim)
        # self.act = nn.GELU()
        # # self.act = nn.LeakyReLU(0.1, inplace=True)
        # self.pwconv2 = nn.Linear(3 * dim, dim)
        # self.pwconv0 = ConvNorm1d(dim, dim, kernel_size=1, stride=1, dilation=0,
        #                               groups=1, norm_layer='bn_1d', act_layer='silu', inplace=True)
        # self.pwconv10 = nn.Conv1d(dim, dim, 1)
        # self.pwconv11 = nn.Conv1d(dim, self.expan_ratio * dim, 5, padding=2, groups=dim)
        # self.conv_spatial = nn.Conv1d(self.expan_ratio * dim, dim, 7, stride=1, padding=3, groups=dim, dilation=1)
        # self.pwconv12 = nn.Conv1d(dim, dim, 1)
        # self.act = nn.LeakyReLU(0.1, inplace=True)


        self.pwconv1 = nn.Conv1d(dim, dim*self.expan_ratio, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pwconv2 = nn.Conv1d(dim*self.expan_ratio, dim, 1)
        self.drop = nn.Dropout(drop_path)
        # self.dwconv = nn.Conv1d(dim*self.expan_ratio, dim*self.expan_ratio, 3, 1, 1, bias=True, groups=dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # print('in', input.shape)
        
        xt = self.ddwconv(x)
        x = self.bn1(xt)
        # x = self.act(x)
        # x = self.drop(x) + xt

        # xi = self.pwconv0(x)
        #
        # x = xi.permute(0, 2, 1)  # (N, C, H, W) -> (N, H, W, C)

        # x = x.permute(0, 2, 1).contiguous()  # (N, C, L) -> (N, L, C)
        # x = self.pwconv1(x)

        # x = self.act(x)
        # x = self.pwconv2(x)

        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.pwconv2(x)

        ## !!!!
        # x_s = self.pwconv10(x)
        # x_s = self.pwconv11(x_s)
        # x_s = self.conv_spatial(x_s)
        # x_s = self.act(x_s)
        # x_s = x_s*input
        # x = self.pwconv12(x_s)


        # x_s = self.bn1(x_s)
        # x_c = x_s.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)

        # x_c = self.pwconv1(x_s)
        # x_c = self.dwconv(x_c)
        # x_c = self.act(x_c)
        # x = self.pwconv2(x_c)

        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1).contiguous()  # (N, L, C) -> (N, C, L)
        # print(x.shape)

        x = input + self.drop_path(x)

        return x

class LGFI1d(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        self.expan_ration = 1
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier1d(dim=self.dim)
        self.temp_embd = TimeEmbeddingSine(11, 1)

        self.norm_xca = networks.depth_encoder.LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = networks.depth_encoder.XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # self.norm = networks.depth_encoder.LayerNorm(self.dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)

        self.pwconv0 = ConvNormAct1d(self.dim, self.dim*self.expan_ration, kernel_size=3, stride=1, dilation=1,
                                      groups=1, norm_layer='bn_1d', act_layer='silu', inplace=True)
        # self.pwconv1 = ConvNormAct1d(self.dim, self.dim*self.expan_ration, kernel_size=1, stride=1, dilation=1,
        #                              groups=1, norm_layer='bn_1d', act_layer='silu', inplace=True)
        # self.proj_drop = nn.Dropout(drop)
        # self.proj = ConvNormAct1d(self.dim*self.expan_ration, self.dim, kernel_size=1, groups=1, norm_layer='bn_1d', act_layer='none', inplace=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        input_ = x
        # print(x.shape)

        # XCA
        B, C, L = x.shape
        # x = x.reshape(B, C, L).permute(0, 2, 1).contiguous()

        if self.pos_embd:
            # pos_encoding = self.pos_embd(B, L)
            # x = x.permute(0, 2, 1) + pos_encoding
            pos_encoding = self.pos_embd(x)
            pos_encoding = torch.mean(pos_encoding, dim=2, keepdim=True)
            # print('pos',pos_encoding.shape)
            # print(pos_encoding.shape)
            # x = x + pos_encoding

            tmp_encoding = self.temp_embd(11)
            # print(tmp_encoding.shape)
            #
            # tmp_encoding = torch.mean(encoding, dim=1, keepdim=True)
            # print(tmp_encoding.shape)
            # x = x + pos_encoding
            x = x + pos_encoding + tmp_encoding.permute(1, 2, 0)
            # print(tmp_encoding.shape)

        x = x.permute(0, 2, 1)

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # x = x.reshape(B, L, C)
        # x = x.reshape(B, C, L)
        # Inverted Bottleneck
        # x = self.norm(x)
        #
        # x = self.pwconv1(x)
        #
        # x = self.act(x)
        # x = self.pwconv2(x)
        x_ = self.pwconv0(x)
        x = x + (x_)
        # x = self.pwconv1(x)

        # x = self.proj_drop(x)
        # x = self.proj(x)

        # x = x.reshape(B, C, L)
        x = x.permute(0, 2, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1).contiguous()  # (N, L, C) -> (N, C, L)

        x = input_ + self.drop_path(x)

        return x


class LGFI1dc(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        # self.pos_embd = None
        self.use_pos_emb = use_pos_emb
        # if use_pos_emb:
        self.pos_embd = PositionalEncodingFourier1d(dim=self.dim)
        # self.temp_embd = TimeEmbeddingSine(21, 1)

        self.norm_xca = networks.depth_encoder.LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = networks.depth_encoder.XCAC(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # self.norm = networks.depth_encoder.LayerNorm(self.dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)

        self.pwconv0 = ConvNormAct1d(self.dim, self.dim, kernel_size=3, stride=1, dilation=1,
                                      groups=1, norm_layer='bn_1d', act_layer='silu', inplace=True)
        # self.pwconv1 = ConvNormAct1d(self.dim, self.dim, kernel_size=1, stride=1, dilation=1,
#                                      groups=1, norm_layer='bn_1d', act_layer='silu', inplace=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ## temproal
        self.seq_len = 11
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
        B, C, L = x.shape
        # x = x.reshape(B, C, L).permute(0, 2, 1).contiguous()

        if self.use_pos_emb:
            # pos_encoding = self.pos_embd(B, L)
            # x = x.permute(0, 2, 1) + pos_encoding
            pos_encoding = self.pos_embd(x)
            # pos_encoding = torch.mean(pos_encoding, dim=2, keepdim=True)

            # tmp_encoding = self.temp_embd(11)

            # x = x + pos_encoding + tmp_encoding.permute(1, 2, 0)

            x = x + pos_encoding

        x = x.permute(0, 2, 1)
        # print(x.shape)

        T = self.seq_len
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)].reshape(T, T, -1)  # T, T, nH
        relative_position_bias = relative_position_bias[None].permute(3, 0, 1, 2).expand(self.num_heads,
                                                                                         (C // self.num_heads) * (
                                                                                                     C // self.num_heads),
                                                                                         T, T)
        relative_position_bias = F.interpolate(relative_position_bias, scale_factor=1/11)
        # print(relative_position_bias.shape)
        relative_position_bias = F.pixel_shuffle(relative_position_bias,
                                                 upscale_factor=int(C // self.num_heads)).permute(1, 0, 2, 3)
        # T = C//self.num_heads
        # relative_position_bias = self.relative_position_bias_table[
        #     self.relative_position_index[:T, :T].reshape(-1)].reshape(T, T, -1)  # T, T, nH
        #
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # relative_position_bias = relative_position_bias.unsqueeze(0)
        # print(relative_position_bias)

        # pos_position = self.pos_embd(1, C//self.num_heads, C//self.num_heads)
        # pos_position = torch.mean(pos_position, dim=1, keepdim=True)
        # # print(pos_position.shape)
        # relative_position_bias = torch.matmul(relative_position_bias, pos_position)
        # # print(relative_position_bias.shape)

        # print(relative_position_bias.shape)
        # print('attn', x.shape)
        # print('tem', relative_position_bias.shape)

        x = x + self.gamma_xca * self.xca(self.norm_xca(x), relative_position_bias, self.use_pos_emb)
        # x = x + self.gamma_xca * self.xca(self.norm_xca(x))
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # x = x.reshape(B, L, C)
        # x = x.reshape(B, C, L)
        # Inverted Bottleneck
        # x = self.norm(x)
        #
        # x = self.pwconv1(x)
        #
        # x = self.act(x)
        # x = self.pwconv2(x)
        x_ = self.pwconv0(x)
        x = x + x_
        # x = self.pwconv1(x)
        # x = x.reshape(B, C, L)
        x = x.permute(0, 2, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1).contiguous()  # (N, L, C) -> (N, C, L)

        x = input_ + self.drop_path(x)

        return x



# The inertial encoder for raw imu data
# class InertialEncoder(nn.Module):
#     def __init__(self, opt):
#         super(InertialEncoder, self).__init__()

#         if model == 'lite-mono':
#             self.num_ch_enc = np.array([48, 80, 128])
#             self.imu = [2, 2, 4]
#             self.dims = [64, 128, 256]
#             if height == 192 and width == 640:
#                 self.dilation = [[1], [1], [1, 2, 3]]
#             elif height == 320 and width == 1024:
#                 self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

#         elif model == 'lite-mono-small':
#             self.num_ch_enc = np.array([48, 80, 128])
#             self.imu = [4, 4, 7]
#             self.dims = [48, 80, 256]
#             if height == 192 and width == 640:
#                 self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
#             elif height == 320 and width == 1024:
#                 self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

#         elif model == 'lite-mono-tiny':
#             self.num_ch_enc = np.array([32, 64, 128])
#             self.imu = [4, 4, 7]
#             self.dims = [32, 64, 256]
#             if height == 192 and width == 640:
#                 self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
#             elif height == 320 and width == 1024:
#                 self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

#         elif model == 'lite-mono-8m':
#             self.num_ch_enc = np.array([64, 128, 224])
#             self.imu = [4, 4, 10]
#             self.dims = [64, 128, 256]
#             if height == 192 and width == 640:
#                 self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
#             elif height == 320 and width == 1024:
#                 self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

#         dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.pose))]

#         self.stages = nn.ModuleList()
#         dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.pose))]
#         cur = 0
#         for i in range(3):
#             stage_blocks = []
#             for j in range(self.pose[i]):
#                 if j > self.pose[i] - global_block[i] - 1:
#                     if global_block_type[i] == 'LGFI':
#                         stage_blocks.append(networks.depth_encoder.LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
#                                                  expan_ratio=expan_ratio,
#                                                  use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
#                                                  layer_scale_init_value=layer_scale_init_value,
#                                                  ))

#                     else:
#                         raise NotImplementedError
#                 else:
#                     stage_blocks.append(networks.depth_encoder.DilatedConv1d(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
#                                                     layer_scale_init_value=layer_scale_init_value,
#                                                     expan_ratio=expan_ratio))

#             self.stages.append(nn.Sequential(*stage_blocks))
#             cur += self.imu[i]

#         self.conv1 = nn.Conv1d(self.dims[0], self.dims[1], kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(self.dims[1], self.dims[2], kernel_size=3, padding=1)

#         self.encoder_conv = nn.Sequential(
#             nn.Conv1d(6, 64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(opt.imu_dropout),
#             nn.Conv1d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(opt.imu_dropout),
#             nn.Conv1d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(opt.imu_dropout))
#         self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)

#     def forward(self, x):
#         # x: (N, seq_len, 11, 6)
#         batch_size = x.shape[0]
#         # seq_len = x.shape[1]
#         x = x.view(batch_size, x.size(1), x.size(2))    # x: (N x seq_len, 11, 6)

#         for i in range(0, 3):
#             for s in range(len(self.stages[i]) - 1):
#                 x = self.stages[i][s](x)
#             x = self.stages[i][-1](x)
                    
#             if i == 0:
#                 x = self.conv1(x)
#                 # print("pose1",x.shape)
#             elif i == 1:
#                 x = self.conv2(x)
#                 # print("pose2",x.shape)

#         print(x.shape)

#         # imu
#         # x = self.encoder_conv(x.permute(0, 2, 1).float())                   # x: (N x seq_len, 64, 11)

#         out = self.proj(x.view(x.shape[0], -1))                             # out: (N x seq_len, 256)
#         return out.view(batch_size, 256)


def initialization(net):
    #Initilization
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



class InertialEncoder(nn.Module):
    def __init__(self, in_chans=6, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI1d', 'LGFI1d', 'LGFI1d'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()

        if model == 'lite-mono':
            self.num_ch_enc = np.array([48, 80, 128])
            self.imu = [1, 1, 2]
            # self.imu = [1, 1, 2]
            self.dims = [64, 128, 224]
            if height == 192 and width == 640:
                self.dilation = [[0], [0], [0]]
                # self.dilation = [[0], [0], [0]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([48, 80, 128])
            self.imu = [4, 4, 7]
            self.dims = [48, 80, 256]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 128])
            self.imu = [4, 4, 7]
            self.dims = [32, 64, 256]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-8m':
            self.num_ch_enc = np.array([64, 128, 224])
            self.imu = [4, 4, 10]
            self.dims = [64, 128, 256]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 11)]

        self.stages = nn.ModuleList()
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.imu))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.imu[i]):
                if j > self.imu[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI1d':
                        stage_blocks.append(LGFI1dc(dim=self.dims[i], drop_path=dp_rates[0],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(DilatedConv1d(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[0],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.imu[i]

        self.conv1 = nn.Conv1d(self.dims[0], self.dims[1], kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(self.dims[1], self.dims[2], kernel_size=3, padding=1, bias=False)
        # self.conv3 = nn.Conv1d(self.dims[2], self.dims[3], kernel_size=1, padding=0)

        # self.encoder_conv = nn.Sequential(
        #     nn.Conv1d(6, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Conv1d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Conv1d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Dropout(0.1))
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0),
            )

        self.proj = nn.Linear(224 * 1 * 11, 256)
        self.drop_out = nn.Dropout(0.1)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        # seq_len = x.shape[1]
        # print(x.shape)
        # x = x.view(batch_size, x.size(1), x.size(2))    # x: (N x seq_len, 11, 6)
        # x = x.permute(0, 2, 1).float()
        x = self.encoder_conv(x.permute(0, 2, 1).float())

        for i in range(0, 3):
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)

            x = self.stages[i][-1](x)
                    
            if i == 0:
                x = self.conv1(x)

            elif i == 1:
                x = self.conv2(x)

            # elif i == 2:
            #     x = self.conv3(x)


        # print(x.shape)
        # x = x.permute(0, 2, 1)

        # imu
        # x = self.encoder_conv(x.permute(0, 2, 1).float())                   # x: (N x seq_len, 64, 11)
        # print(x.shape)

        out = self.proj(x.view(x.shape[0], -1))                             # out: (N x seq_len, 256)
        # out = self.drop_out(out)  
        return out.view(batch_size, 256)