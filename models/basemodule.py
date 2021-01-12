import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelU(nn.Module):

    where = "audio"

    def __init__(self, in_channel, out_channel, is_pool=True):
        super(ConvBNRelU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6()
        if is_pool:
            self.pool = nn.MaxPool2d(kernel_size=2)
        pass

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride, pad):
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)


class ConvBase(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad):
        super(ConvBase, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride, pad),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.main(x)


class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad):
        super(UpConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 2, kernel, stride, pad),
            nn.BatchNorm2d(out_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel * 2, out_channel, kernel, stride, pad),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        return self.main(x)


class Conv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel, stride, pad, bias=True)
        self.bn = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)

# class ResNetBackbone(nn.Module):
#
#     def __init__(self, block, layers, out_dim=1024):
#         self.inplanes = 64
#         super(ResNetBackbone, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # modification: cut a layer
#
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.dropout = nn.Dropout(p=0.5)
#         self.outfc = nn.Linear(1024 * block.expansion, out_dim)  # modification: 512 --> 1024
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         # x = self.layer4(x)  # [N, 2048, 4, 4]
#         print(x.size())
#
#         x = self.avgpool(x)  # size is too small
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         x = self.outfc(x)
#
#         return x
#
#
# class ResNetBottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(ResNetBottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
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
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet50(nn.Module):
#
#     def __init__(self):
#         super(ResNet50, self).__init__()
#         self.resnet50 = ResNetBackbone(ResNetBottleneck, [3, 4, 6, 3])
#
#     def forward(self, x):
#         x = self.resnet50(x)
#         return x
