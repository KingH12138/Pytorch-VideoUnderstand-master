from torch import nn

from models.channel_attention import Channel_Attention
from models.spatial_attention import Spatial_Attention


class CS_CBAM(nn.Module):
    def __init__(self,in_channel, ratio=4, sa_kernel_size=(1,5,5)):
        super(CS_CBAM, self, ).__init__()
        self.ca = Channel_Attention(in_channel, ratio)
        self.sa = Spatial_Attention(sa_kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# import torch
# device = torch.device('cuda')
# ta = CS_CBAM(512).to(device)
# x = torch.rand((16,512,8,26,26)).to(device)
# output = ta(x)
# print(x.shape)


