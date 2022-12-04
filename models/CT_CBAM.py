from torch import nn
from models.temporal_attention import Temporal_Attention
from models.channel_attention import Channel_Attention


class CT_CBAM(nn.Module):
    def __init__(self,in_channel, ratio=4, ta_kernel_size=(5,1,1)):
        super(CT_CBAM, self).__init__()
        self.ta = Temporal_Attention(ta_kernel_size)
        self.ca = Channel_Attention(in_channel, ratio)

    def forward(self, x):
        x = self.ca(x)
        x = self.ta(x)
        return x

# import torch
# device = torch.device('cuda')
# ta = CT_CBAM(512).to(device)
# x = torch.rand((16,512,8,26,26)).to(device)
# output = ta(x)
# print(x.shape)
