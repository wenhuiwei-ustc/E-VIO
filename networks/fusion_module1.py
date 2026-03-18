import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import networks
import networks.depth_encoder
import networks.imu_encoder
from timm.models.layers import DropPath
from networks.basic_modules import ConvNormAct1d, ConvNorm1d

class PositionalEncodingFourier1d(nn.Module):

    def __init__(self, d_hid=1, dim=256):
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


# The fusion module
class FusionModule(nn.Module):
    def __init__(self, opt):
        super(FusionModule, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.i_f_len + opt.v_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        # elif self.fuse_method == 'hard':
        #     self.net = nn.Sequential(
        #         nn.Linear(self.f_len, 2 * self.f_len))

    def forward(self, v, imu):
        if self.fuse_method == 'cat':
            return torch.cat((v, imu), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, imu), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        # elif self.fuse_method == 'hard':
        #     feat_cat = torch.cat((v, imu), -1)
        #     weights = self.net(feat_cat)
        #     weights = weights.view(v.shape[0], self.f_len, 2)
        #     mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
        #     return feat_cat * mask[:, :, 0]

# The policy network module
class PolicyNet(nn.Module):
    def __init__(self, opt):
        super(PolicyNet, self).__init__()
        in_dim = opt.rnn_hidden_size + opt.i_f_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2))

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return logits, hard_mask


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

class FuseModule(nn.Module):
    def __init__(self, channels, reduction):
        super(FuseModule, self).__init__()
        self.fc1 = nn.Linear(channels, channels * reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels * reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        B, H, W = x.shape
        x = x.view(B, -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(B, H, W)
        return module_input * x

class Trans_Fusion(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        self.expan_ration = 1
        if use_pos_emb:
            self.pos_embdx = PositionalEncodingFourier1d(dim=self.dim)
            self.pos_embdy = PositionalEncodingFourier1d(dim=256)

        self.norm_xca = networks.depth_encoder.LayerNorm(self.dim, eps=1e-6)
        self.norm_xcay = networks.depth_encoder.LayerNorm(256, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = networks.depth_encoder.XCAt(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # self.norm =  networks.depth_encoder.LayerNorm(self.dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.pwconv0 = ConvNormAct1d(self.dim, self.dim, kernel_size=3, stride=1, dilation=1,
                                      groups=1, norm_layer='bn_1d', act_layer='silu', inplace=True)
        # self.proj_drop = nn.Dropout(drop)
        # self.proj = ConvNormAct1d(self.dim*self.expan_ration, self.dim, kernel_size=1, groups=1, norm_layer='bn_1d', act_layer='none', inplace=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    #     ## temproal
    #     self.seq_len = 1
    #     self.num_heads = num_heads
    #     self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * self.seq_len - 1, self.num_heads))  # 2*T-1, nH
    #     self.register_buffer("relative_position_index", self.get_position_index())  # T, T
    #
    # def get_position_index(self):
    #     coords = torch.arange(self.seq_len)  # T
    #     relative_coords = coords[:, None] - coords[None, :]  # T, T
    #     relative_coords += self.seq_len - 1  # shift to start from 0
    #     relative_position_index = relative_coords  # T, T
    #     return relative_position_index

    def forward(self, x, y):
        input_ = x

        # XCA
        B, L, C = x.shape
        # x = x.reshape(B, C, L).permute(0, 2, 1).contiguous()

        if self.pos_embdx:
            # pos_encoding = self.pos_embd(B, L)
            # x = x.permute(0, 2, 1) + pos_encoding
            pos_encoding = self.pos_embdx(x)
            # print(pos_encoding.shape)
            # pos_encoding = torch.mean(pos_encoding, dim=2, keepdim=True)
            x = x + pos_encoding

            pos_encodingy = self.pos_embdx(y)

            y = y + pos_encodingy

        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        # T = self.seq_len
        # relative_position_bias = self.relative_position_bias_table[
        #     self.relative_position_index.reshape(-1)].reshape(T, T, -1)  # T, T, nH
        # relative_position_bias = relative_position_bias[None].permute(3, 0, 1, 2).expand(self.num_heads, (C)*(C), T, T)
        # relative_position_bias = F.pixel_shuffle(relative_position_bias, upscale_factor=C).permute(1, 0, 2, 3)
        # x = x + self.gamma_xca * self.xca(self.norm_xca(x), relative_position_bias)
        x = y + self.gamma_xca * self.xca(self.norm_xca(x), self.norm_xcay(y))
        # x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        # x = x.reshape(B, L, C)
        x = x.permute(0, 2, 1).contiguous()
        # Inverted Bottleneck
        # x = self.norm(x)
        #
        # x = self.pwconv1(x)
        #
        # x = self.act(x)
        # x = self.pwconv2(x)
        x_ = self.pwconv0(x)
        x = x + x_

        # x = self.proj_drop(x)
        # x = self.proj(x)

        x = x.permute(0, 2, 1).contiguous()
        # x = x.reshape(B, C, L)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1).contiguous()  # (N, L, C) -> (N, C, L)

        x = input_ + self.drop_path(x)

        return x

