import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet, vgg


def get_norm(name, c_out):
    if name == 'batch':
        norm = nn.BatchNorm2d(c_out)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(c_out)
    else:
        norm = None
    return norm


def get_act(name):
    if name == 'relu':
        activation = nn.ReLU()
    elif name == 'elu':
        activation == nn.ELU()
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation


class ConvBlock(nn.Module):

    def __init__(
            self, c_in, c_out, s_kernel, stride, norm=None, act=None):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        layers = []
        layers.append(
            self.get_padded_conv(self.c_in, self.c_out, s_kernel, stride))
        if norm:
            layers.append(get_norm(norm, self.c_out))
        if act:
            layers.append(get_act(act))
        self.encoder = nn.Sequential(*layers)

    def get_padded_conv(self, c_in, c_out, s_kernel, stride):
        if (s_kernel - stride) % 2 == 0:
            s_pad =  (s_kernel - stride) // 2
            conv = nn.Conv2d(c_in, c_out, s_kernel, stride, s_pad)
        else:
            left = (s_kernel - stride) // 2
            right = left + 1
            conv = nn.Sequential(
                nn.ConstantPad2d((left, right, left, right), 0),
                nn.Conv2d(c_in, c_out, s_kernel, stride, 0)
            )
        return conv

    def forward(self, x):
        return self.encoder(x)


class ConvBlocks(nn.Module):

    def __init__(
            self, c_in=3, c_kernels=[16, 32, 64, 64], s_kernels=[5, 3, 3, 3],
            strides=2, norm='batch', act='relu'):
        super().__init__()

        if type(strides) is not list:
            strides = [strides] * len(c_kernels)
        assert len(c_kernels) == len(s_kernels) == len(strides), (
            'The length of channel ({}), size ({}) and stride ({}) must equal'
            '.'.format(len(c_kernels), len(s_kernels), len(strides)))

        self.c_in = [c_in] + c_kernels[:-1]
        self.c_out = c_kernels
        self.s_kernels = s_kernels
        self.strides = strides
        self.norm = norm
        self.act = act

        layers = []
        for i, (c_in, c_out, s_kernel, stride) in enumerate(
                zip(self.c_in, self.c_out, self.s_kernels, self.strides)):
            if i == 0 or i == (len(self.c_in) - 1):
                layer_norm = layer_act = None
            else:
                layer_norm, layer_act = self.norm, self.act
            layers.append(ConvBlock(
                c_in, c_out, s_kernel, stride, layer_norm, layer_act))

        self.cnn = nn.Sequential(*layers)

    def forward(self, imgs):
        return self.cnn(imgs)


class FCBlock(nn.Module):

    def __init__(self, c_in, c_out, act=None, dropout=None):
        super().__init__()
        layers = [nn.Linear(c_in, c_out)]
        if act:
            layers.append(get_act(act))
        if dropout:
            dropout = 0.5 if dropout is True else dropout
            layers.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class FCBlocks(nn.Module):

    def __init__(self, c_in, c_out, act=None, dropout=None):
        super().__init__()

        self.c_in = [c_in] + c_out[:-1]
        self.c_out = c_out
        self.act = act
        self.dropout = dropout

        layers = []
        for i, (c_in, c_out) in enumerate(zip(self.c_in, self.c_out)):
            act = None if i == (len(self.c_out) - 1) else self.act
            dropout = None if i == (len(self.c_out) - 1) else self.dropout
            layers.append(FCBlock(c_in, c_out, act=act, dropout=dropout))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class ResNetBlock(nn.Module):

    def __init__(
            self, arch='resnet18', avgpool=True, pretrained=False, c_in=3):
        super().__init__()

        if c_in != 3:
            pretrained = False

        model_func = getattr(resnet, arch)
        model = model_func(pretrained=pretrained)

        if c_in != 3:
            inplanes = 64
            conv1 = nn.Conv2d(
                c_in, inplanes, kernel_size=7, stride=2, padding=3,
                bias=False)
            nn.init.kaiming_normal_(
                conv1.weight, mode='fan_out', nonlinearity='relu')
        else:
            conv1 = model.conv1

        layers = [
            conv1, model.bn1,
            model.relu, model.maxpool,
            model.layer1, model.layer2,
            model.layer3, model.layer4
        ]
        if avgpool:
            layers.append(model.avgpool)

        self.encoder = nn.Sequential(*layers)

        if 'Bottleneck' in str(type(model.layer4[-1])):
            self.c_out = model.layer4[-1].conv3.out_channels
        else:
            self.c_out = model.layer4[-1].conv2.out_channels

    def forward(self, imgs):
        return self.encoder(imgs)


class VGGBlock(nn.Module):  

    def __init__(self, arch='vgg11_bn', pretrained=False):
        super().__init__()

        model_func = getattr(vgg, arch)
        model = model_func(pretrained=pretrained)

        self.encoder = nn.Sequential(*list(model.features.children())[:-1])
        for layer in model.features[::-1]:
            if hasattr(layer, 'out_channels'):
                self.c_out = layer.out_channels
                break

    def forward(self, imgs):
        return self.encoder(imgs)
