import torch
from torch import nn


class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=(1,5,5)):
        super(Spatial_Attention, self).__init__()
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=(0,kernel_size[-1]//2,kernel_size[-1]//2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b,c,t,h,w -> b,1,1,h,w
        max_w, _ = torch.max(x, dim=1, keepdim=True)
        max_w, _ = torch.max(max_w, dim=2, keepdim=True)
        # b,c,t,h,w -> b,1,1,h,w
        avg_w = torch.mean(x, dim=1, keepdim=True)
        avg_w = torch.mean(avg_w, dim=2, keepdim=True)
        # b,1,1,h,w -> b,2,1,h,w
        w = torch.cat([avg_w, max_w], dim=1)
        # b,2,1,h,w -> b,1,1,h,w
        w = self.conv(w)
        w = self.sigmoid(w)

        return w*x

