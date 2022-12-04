from torch import nn
from models.temporal_attention import Temporal_Attention
from models.channel_attention import Channel_Attention
from models.spatial_attention import Spatial_Attention


class CBAM_3D(nn.Module):
    def __init__(self,in_channel, ratio=4, ta_kernel_size=(5,1,1),sa_kernel_size=(1,5,5)):
        super(CBAM_3D, self).__init__()
        self.ta = Temporal_Attention(ta_kernel_size)
        self.ca = Channel_Attention(in_channel, ratio)
        self.sa = Spatial_Attention(sa_kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        x = self.ta(x)

        return x

# import torch
# device = torch.device('cuda')
# ta = CBAM_3D(512).to(device)
# x = torch.rand((16,512,8,26,26)).to(device)
# output = ta(x)
# print(x.shape)