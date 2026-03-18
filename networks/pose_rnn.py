import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda
import networks
from collections import OrderedDict

import torch.nn.init as init


class PoseNet(nn.Module):
    '''
    Fuse both features and output the 6 DOF camera pose
    '''
    def __init__(self, input_size=1024):
        super(PoseNet, self).__init__()
        self.se = networks.FuseModule(input_size, 16)
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=1024,
                           num_layers=2,
                           batch_first=True)

        self.fc1 = nn.Sequential(nn.Linear(1024, 6))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                init.xavier_normal_(m.all_weights[0][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[0][1], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][1], gain=np.sqrt(1))
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data, gain=np.sqrt(1))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, visual_fea, imu_fea):
        self.rnn.flatten_parameters()
        inpt = torch.cat((visual_fea, imu_fea), dim=-1)
        # print(inpt.shape)
        B, L = inpt.shape

        inpt = inpt.view(B, L, 1)
        inpt = self.se(inpt)
        _, MM, _ = inpt.shape
        inpt = inpt.view(B, 1, MM)
        out, (h, c) = self.rnn(inpt)
        out = 0.01 * self.fc1(out).view(-1, 1, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        # print(axisangle.shape)
        return axisangle, translation

class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network

        f_len = opt.v_f_len
        # f_len = opt.v_f_len
        
        # self.rnn = nn.LSTM(
        #     input_size=f_len,
        #     hidden_size=opt.rnn_hidden_size,
        #     num_layers=2,
        #     dropout=opt.rnn_dropout_between,
        #     batch_first=True)
        stride = 1
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv1d(f_len, 512, 1)
        self.convs[("pose", 0)] = nn.Conv1d(512, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv1d(256, 256, 1, stride, 1)
        self.convs[("pose", 2)] = nn.Conv1d(256, 6*2, 1)

        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.net = nn.ModuleList(list(self.convs.values()))

        # self.fuse = networks.FusionModule(opt)
        # self.fuse = networks.Trans_Fusion(dim=opt.v_f_len)
        # self.fuse = networks.FuseModule(channels = f_len, reduction=4)

        # The output networks
        # self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        # self.regressor = nn.Sequential(
        #     nn.Linear(opt.rnn_hidden_size, 128),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(128, 6))

    # def forward(self, fv, fv_alter, fi, dec, prev=None):
    #     if prev is not None:
    #         prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
    #     # Select between fv and fv_alter
    #     v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv

    #     # print(v_in.shape)
    #     # print(fi.shape)

    #     # trans_fusion_module
    #     fused = torch.cat((fv, fi), -1)
    #     B, L = fused.shape

    #     fused = fused.view(B, L, 1)
    #     # print(fused.shape)
    #     fused = self.fuse(fused)

    #     # fused = self.fuse(fv, fi)
    #     # fused = fused.view(len(fused), 1, -1)

    #     # fv = fv.view(len(fv), 1, -1)
        
    #     # out, hc = self.rnn(fv)
    #     # out, hc = self.rnn(fused)
    #     # out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)

    #     # ours
    #     pose = self.decoder_conv1d(fused)

    #     # out = self.rnn_drop_out(out)
    #     # pose = self.regressor(out)
    #     # pose = pose.view(-1, 2, 1, 6)


    #     # hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
    #     # return pose, hc
    #     return pose
    #     # return axisangle, translation, hc

    def forward(self, fused):
        # print(fv.shape)
        # print(fi.shape)

        # trans_fusion_module
        # fused = torch.cat((fv, fi), -1)
        # B, L = fused.shape
        
        # fused = fused.view(B, L, 1)
        # fused = self.fuse(fused)

        # cross fusion
        # Bv, Lv = fv.shape
        # fv = fv.view(Bv, Lv, 1)
        # Bi, Li = fi.shape
        # fi = fi.view(Bi, Li, 1)   
        # fused = self.fuse(fv, fi)                                                                                      

        # Bv, Lv = fv.shape
        # fv = fv.view(Bv, Lv, 1)
        # Bi, Li = fi.shape
        # fi = fi.view(Bi, Li, 1)
        # fused = self.fuse(fv, fi)


        # fused = self.fuse(fv, fi)
        # fused = fused.view(len(fused), 1, -1)

        # fv = fv.view(len(fv), 1, -1)
        
        # out, hc = self.rnn(fv)
        # out, hc = self.rnn(fused)
        # out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)

        cat_features = self.relu(self.convs["squeeze"](fused))

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(2)

        out = 0.01 * out.view(-1, 2, 1, 6)
        # print(out.shape)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

        # # ours
        # pose = self.decoder_conv1d(fused)
        # print(pose.shape)

        # out = self.rnn_drop_out(out)
        # pose = self.regressor(out)
        # pose = pose.view(-1, 2, 1, 6)


        # hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        # return pose, hc
        # return pose
        # return axisangle, translation, hc
