import torch
import torch.nn.functional as F
from torch import nn

from .block import ConvBlocks, ResNetBlock, VGGBlock


class BaseEncoder(nn.Module):

    def __init__(self, encoder, mode):
        super().__init__()
        self.encoder = encoder
        self.mode = mode

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        tensor = torch.zeros(batch, channel, height, width).to(
            next(self.encoder.parameters()).device)
        _, c_out, h_out, w_out = tuple(self.forward(tensor, tensor).shape)
        return c_out, h_out, w_out

    def forward(self, init, fin):
        if self.mode == 'concat':
            out = torch.cat([init, fin], dim=1)
        elif self.mode == 'subtract':
            out = init - fin
        else:
            raise NotImplementedError
        out = self.encoder(out)
        return out


class CNNEncoder(BaseEncoder):

    def __init__(
            self, mode, c_kernels=[16, 32, 64, 64], s_kernels=[5, 3, 3, 3]):
        c_in = 3 if mode == 'subtract' else 6
        encoder = ConvBlocks(
            c_in=c_in, c_kernels=c_kernels, s_kernels=s_kernels)
        super().__init__(encoder, mode)


class ResNetEncoder(BaseEncoder):
    def __init__(self, mode, arch="resnet18"):
        c_in = 3 if mode == 'subtract' else 6
        encoder = ResNetBlock(arch, c_in=c_in)
        super().__init__(encoder, mode)


class DUDAEncoder(nn.Module):

    def __init__(self, arch):
        super().__init__()

        self.encoder = ResNetBlock(arch)
        c_out = self.encoder.c_out

        self.attention = nn.Sequential(
            ConvBlocks(
                c_out * 2, c_kernels=[c_out, c_out], s_kernels=[3, 3],
                strides=1, norm=None, act='relu'),
            nn.Sigmoid()
        )

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        tensor = torch.zeros(batch, channel, height, width).to(
            next(self.encoder.parameters()).device)
        _, c_out, h_out, w_out = tuple(self.forward(tensor, tensor).shape)
        return c_out, h_out, w_out

    def forward(self, init, fin):

        x_init = self.encoder(init)
        x_fin = self.encoder(fin)
        x_diff = x_fin - x_init

        a_init = self.attention(torch.cat([x_init, x_diff], dim=1))
        a_fin = self.attention(torch.cat([x_fin, x_diff], dim=1))

        l_init = x_init * a_init
        l_fin = x_fin * a_fin
        l_diff = l_fin - l_init

        res = torch.cat([l_init, l_fin, l_diff], dim=1)

        return res


class BCNNEncoder(nn.Module):

    def __init__(self, arch='vgg11_bn'):
        super().__init__()
        self.encoder = VGGBlock(arch)

    def get_output_shape(self, height, width):
        batch, channel = 1, 3
        tensor = torch.zeros(batch, channel, height, width).to(
            next(self.encoder.parameters()).device)
        _, c_out = tuple(self.forward(tensor, tensor).shape)
        return c_out

    def forward(self, init, fin):
        init = self.encoder(init)
        N, C, H, W = init.shape
        init = init.view(N, C, H * W)
        L = init.shape[-1]

        fin = self.encoder(fin)
        fin = fin.view(N, C, H * W).transpose(1, 2)

        res = torch.bmm(init, fin) / L

        res = res.view(N, -1)
        res = torch.sqrt(res + 1e-5)
        res = F.normalize(res)

        return res