import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda
from depth_pose_prediction.networks.depth_encoder import *

from torch.autograd import Variable


class PoseEncoder(nn.Module):
    """
    Pose_Encoder
    """

    def __init__(self, in_chans=6, model='SECT', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()
        if model == 'SECT':
            self.num_ch_enc = np.array([48, 80, 128])
            self.pose = [2, 2, 4]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1], [1], [1, 2, 3]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']

        self.stem = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.pose))]
        cur = 0
        self.mlp_ratios = [2, 2, 2]
        self.sr_ratios = [4, 2, 1]
        for i in range(3):
            stage_blocks = []
            for j in range(self.pose[i]):
                if j > self.pose[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))
                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(GatedCNNBlock(dim=self.dims[i],
                                                      dilation=self.dilation[i][j], kernel_size=3,
                                                      drop_path=dp_rates[cur + j],
                                                      ))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.pose[i]

        self.conv1 = Conv(self.dims[0], self.dims[1], kSize=3, stride=2, padding=1, bn_act=False)
        self.conv2 = Conv(self.dims[1], self.dims[2], kSize=3, stride=2, padding=1, bn_act=False)
        self.conv3 = Conv(self.dims[2], self.dims[2], kSize=3, stride=2, padding=1, bn_act=False)
        
        self.drop_out = nn.Dropout(0.25)
       
        self.apply(self._init_weights)

    def forward_features(self, x):
        
        x = (x - 0.45) / 0.225
        x = self.stem(x)

        for i in range(0, 3):
            input_ = x
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
            x = x.reshape(input_.shape)

            if i == 0:
                x = self.conv1(x)
            elif i == 1:
                x = self.conv2(x)
            elif i == 2:
                x = self.conv3(x)

        features = x

        return features

    def forward(self, x):
        x = self.forward_features(x)
        # print(x.shape)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
