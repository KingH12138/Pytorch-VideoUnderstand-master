from torch import nn


class Channel_Attention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(Channel_Attention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool3d(output_size=1)

        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.shape
        max_pool = self.max_pool(x) # b,c,t,h,w -> b,c,1,1,1
        avg_pool = self.avg_pool(x)  # b,c,t,h,w -> b,c,1,1,1

        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])
        # print(max_pool.shape)   #########################################################################################
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)
        w = x_maxpool + x_avgpool
        w = self.sigmoid(w)
        w = w.view([b, c, 1, 1, 1])

        return w*x