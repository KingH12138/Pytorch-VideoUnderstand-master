# from functools import partial
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.CT_CBAM import CT_CBAM
#
#
# def get_inplanes():
#     return [64, 128, 256, 512]
#
#
# def conv3x3x3(in_planes, out_planes, stride=1):
#     return nn.Conv3d(in_planes,
#                      out_planes,
#                      kernel_size=3,
#                      stride=stride,
#                      padding=1,
#                      bias=False)
#
#
# def conv1x1x1(in_planes, out_planes, stride=1):
#     return nn.Conv3d(in_planes,
#                      out_planes,
#                      kernel_size=1,
#                      stride=stride,
#                      bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super().__init__()
#
#         self.conv1 = conv3x3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.ct = CT_CBAM(planes)  ############################################################################################
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out = self.ct(out)  ############################################################################################
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super().__init__()
#
#         self.conv1 = conv1x1x1(in_planes, planes)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = conv3x3x3(planes, planes, stride)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = conv1x1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm3d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.ct = CT_CBAM(planes * self.expansion)  ############################################################################################
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out = self.ct(out)  ############################################################################################
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self,
#                  block,
#                  layers,
#                  block_inplanes,
#                  n_input_channels=3,
#                  conv1_t_size=7,
#                  conv1_t_stride=1,
#                  no_max_pool=False,
#                  shortcut_type='B',
#                  widen_factor=1.0,
#                  n_classes=4):
#         super().__init__()
#
#         block_inplanes = [int(x * widen_factor) for x in block_inplanes]
#
#         self.in_planes = block_inplanes[0]
#         self.no_max_pool = no_max_pool
#
#         self.conv1 = nn.Conv3d(n_input_channels,
#                                self.in_planes,
#                                kernel_size=(conv1_t_size, 7, 7),
#                                stride=(conv1_t_stride, 2, 2),
#                                padding=(conv1_t_size // 2, 3, 3),
#                                bias=False)
#         self.bn1 = nn.BatchNorm3d(self.in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
#         ##############################################################################################################
#         params = torch.ones(size=(4, 1, 1), requires_grad=True)  # 初始化断章机制的权重矩阵
#         self.weight_matrix = nn.Parameter(params)
#         ##############################################################################################################
#         self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
#                                        shortcut_type)
#         self.layer2 = self._make_layer(block,
#                                        block_inplanes[1],
#                                        layers[1],
#                                        shortcut_type,
#                                        stride=2)
#         self.layer3 = self._make_layer(block,
#                                        block_inplanes[2],
#                                        layers[2],
#                                        shortcut_type,
#                                        stride=2)
#         self.layer4 = self._make_layer(block,
#                                        block_inplanes[3],
#                                        layers[3],
#                                        shortcut_type,
#                                        stride=2)
#
#         # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         # self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
#         ###############################################################################################################
#         self.trans1 = TransModule(64, n_classes)
#         self.trans2 = TransModule(128, n_classes)
#         self.trans3 = TransModule(256, n_classes)
#         self.trans4 = TransModule(512, n_classes)
#         ###############################################################################################################
#         self.soft = nn.Softmax(dim=1)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight,
#                                         mode='fan_out',
#                                         nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _downsample_basic_block(self, x, planes, stride):
#         out = F.avg_pool3d(x, kernel_size=1, stride=stride)
#         zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
#                                 out.size(3), out.size(4))
#         if isinstance(out.data, torch.cuda.FloatTensor):
#             zero_pads = zero_pads.cuda()
#
#         out = torch.cat([out.data, zero_pads], dim=1)
#
#         return out
#
#     def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
#         downsample = None
#         if stride != 1 or self.in_planes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(self._downsample_basic_block,
#                                      planes=planes * block.expansion,
#                                      stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     conv1x1x1(self.in_planes, planes * block.expansion, stride),
#                     nn.BatchNorm3d(planes * block.expansion))
#
#         layers = []
#         layers.append(
#             block(in_planes=self.in_planes,
#                   planes=planes,
#                   stride=stride,
#                   downsample=downsample))
#         self.in_planes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.in_planes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         if not self.no_max_pool:
#             x = self.maxpool(x)
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#         ##############################################################################################################
#         x1 = self.trans1(x1).unsqueeze(0)
#         x2 = self.trans2(x2).unsqueeze(0)
#         x3 = self.trans3(x3).unsqueeze(0)
#         x4 = self.trans4(x4).unsqueeze(0)
#         ##########  concat  and weighted #############
#         outputs = torch.cat([x1, x2, x3, x4], dim=0)  # 4,1,b,4->4,b,4
#         outputs = outputs * self.weight_matrix  # 4*b*4 times 4*1*1 = 4*b*4
#         outputs = outputs.sum(dim=0)   # 4,b,4 -> b,4
#         ##############################################################################################################
#         return outputs
#
#
# class TransModule(nn.Module):
#     def __init__(self, inplanes, num_class=4):
#         """
#         :param inplanes:in_channel
#         :param num_class: number of classes
#
#         """
#         super(TransModule, self).__init__()
#         transformer_layer = nn.TransformerEncoderLayer(
#             d_model=inplanes,
#             nhead=16,
#             dropout=0.5,
#             activation='relu'
#         )
#         self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
#         self.fc1_linear = nn.Linear(inplanes, num_class)
#         self.soft = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         """
#         inputs:tensor(b,c,t,h,w)
#         """
#         x = x.mean(-1)
#         x = x.mean(-1)
#         # b,c,t,h,w->b,c,t
#         b,_,_ = x.shape
#         x = x.permute(2,0,1) # b,c,t->t(特征帧相当于视频长度，也就是序列长度),b,c
#         x = self.transformer_encoder(x)
#         x = torch.mean(x, dim=0)
#         x = self.fc1_linear(x)
#         x = self.soft(x)
#         return x
#
#
# def generate_model(model_depth, **kwargs):
#     assert model_depth in [10, 18, 34, 50, 101, 152, 200]
#
#     if model_depth == 10:
#         model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
#     elif model_depth == 18:
#         model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
#     elif model_depth == 34:
#         model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 50:
#         model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 101:
#         model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
#     elif model_depth == 152:
#         model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
#     elif model_depth == 200:
#         model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
#
#     return model
#
#
# # device = torch.device('cuda')
# # x = torch.rand((32,3,16,112,112)).to(device)
# # model = generate_model(18, n_classes=4).to(device)
# # y = model(x)



from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) # [224, 224]
        patch_size = (patch_size, patch_size) # [16, 16]
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # [14, 14]
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 14 * 14 = 196
        # 按照划分patch应当是196个[16, 16, 3]的图片块，16*16*3刚好=768，而下一行的conv函数的参数正是这些
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape 切割的思想，不是降维！
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
##############################################################################################
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


class CT_CBAM(nn.Module):
    def __init__(self,in_channel, ratio=4, ta_kernel_size=(5,1,1)):
        super(CT_CBAM, self).__init__()
        self.ta = Temporal_Attention(ta_kernel_size)
        self.ca = Channel_Attention(in_channel, ratio)

    def forward(self, x):
        x = self.ca(x)
        x = self.ta(x)
        return x

##############################################################################################

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.ct = CT_CBAM(planes)  ############################################################################################

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ct(out)  ############################################################################################

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.ct = CT_CBAM(planes * self.expansion)  ############################################################################################

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ct(out)  ############################################################################################
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=4):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        ##############################################################################################################
        params = torch.ones(size=(5, 1, 1), requires_grad=True)  # 初始化断章机制的权重矩阵
        self.weight_matrix = nn.Parameter(params)
        ##############################################################################################################
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        ###############################################################################################################
        self.trans_way = VisionTransformer(in_c=64,num_classes=n_classes,img_size=28,patch_size=7)
        self.fc1 = nn.Linear(64, n_classes, bias=False)
        self.fc2 = nn.Linear(128, n_classes, bias=False)
        self.fc3 = nn.Linear(256, n_classes,bias=False)
        self.fc4 = nn.Linear(512, n_classes,bias=False)
        ###############################################################################################################
        self.soft = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x1 = self.layer1(x)

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        x4 = self.layer4(x3)

        trans_way_inputs = x[:,:,x.shape[2]//2,:,:]
        x5 = self.trans_way(trans_way_inputs)    # tran_way输出中间帧的空间评价分数
        x5 = self.soft(x5)
        x5 = x5.unsqueeze(0)
        ##############################################################################################################
        x1 = self.avgpool(x1)  # b,c,t,h,w->b,c,1,1,1
        x1 = x1.view(x.size(0), -1)  # b,c,1,1,1->b,c
        x1 = self.fc1(x1)  # b,n->b,4
        x1 = self.soft(x1)
        x1 = x1.unsqueeze(0)  # b,4->1,b,4

        x2 = self.avgpool(x2)  # b,c,t,h,w->b,c,1,1,1
        x2 = x2.view(x2.size(0), -1)  # b,c,1,1,1->b,c
        x2 = self.fc2(x2)  # b,n->b,4
        x2 = self.soft(x2)
        x2 = x2.unsqueeze(0)  # b,4->1,b,4

        x3 = self.avgpool(x3)  # b,c,t,h,w->b,c,1,1,1
        x3 = x3.view(x3.size(0), -1)  # b,c,1,1,1->b,c
        x3 = self.fc3(x3)  # b,n->b,4
        x3 = self.soft(x3)
        x3 = x3.unsqueeze(0)  # b,4->1,b,4

        x4 = self.avgpool(x4)  # b,c,t,h,w->b,c,1,1,1
        x4 = x4.view(x.size(0), -1)  # b,c,1,1,1->b,c
        x4 = self.fc4(x4)  # b,n->b,4
        x4 = self.soft(x4)
        x4 = x4.unsqueeze(0)  # b,4->1,b,4

        outputs = torch.cat([x1, x2, x3, x4,x5], dim=0)  # 1,b,4->4,b,4
        outputs = outputs * self.weight_matrix  # 4*b*4 times 4*b*1 = 4*b*4
        outputs = outputs.sum(dim=0).squeeze(0)
        ##############################################################################################################
        return outputs


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


# device = torch.device('cuda')
# x = torch.rand((32,3,16,112,112)).to(device)
# model = generate_model(18, n_classes=4).to(device)
# y = model(x)