"""

时间注意力机制:自创的时间注意力机制可以将更多注意力集中在相关的时间维度上

"""


import torch
from torch import nn


class Temporal_Attention(nn.Module):
    def __init__(self, kernel_size=(5,1,1)):
        """
        :param inplane: 输入的T维度大小
        :param kernel_size: 融合卷积操作的卷积核大小
        """
        super(Temporal_Attention, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=(kernel_size[0]//2,0,0), bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.shape
        # 先生成一个(b,1,c,1,1)的权重
        # b,c,t,h,w->b,1,t,1,1
        max_w, _ = torch.max(x, dim=3, keepdim=True)
        max_w, _ = torch.max(max_w, dim=4, keepdim=True)
        max_w, _ = torch.max(max_w, dim=1, keepdim=True)
        # b,c,t,h,w->b,1,t,1,1
        avg_w = torch.mean(x, dim=3, keepdim=True)
        avg_w = torch.mean(avg_w, dim=4, keepdim=True)
        avg_w = torch.mean(avg_w, dim=1, keepdim=True)
        # b,1,t,1,1->b,2,t,1,1，时域信息融合
        w = torch.cat([avg_w, max_w], dim=1)
        # b,2,t,1,1->b,1,t,1,1
        w = self.conv(w)
        # 归一化
        w = self.sig(w)
        return w*x

# device = torch.device('cuda')
# ta = Temporal_Attention().to(device)
# x = torch.rand((16,512,8,26,26)).to(device)
# output = ta(x)
# print(x.shape)