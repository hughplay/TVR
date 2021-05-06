from torch import nn

from .encoder import CNNEncoder, ResNetEncoder, DUDAEncoder, BCNNEncoder
from .decoder import TransformationDecoder


class SingleTranceNet(nn.Module):

    def __init__(
            self, encoder, height=120, width=160, c_obj=19, n_pair=33,
            fc_img_en=[1024, 256], fc_fusion=[256, 256]):
        super().__init__()
        self.encoder = encoder
        c_out, h_out, w_out = self.encoder.get_output_shape(height, width)
        c_flat = c_out * h_out * w_out
        self.decoder = TransformationDecoder(
            c_flat, c_obj, n_pair, fc_img_en, fc_fusion)

    def forward(self, init, fin, init_desc, obj_target_vec=None):
        img_feat = self.encoder(init, fin)
        img_feat_flat = img_feat.flatten(start_dim=1)
        obj_choice, pair_choice = self.decoder(
            init_desc, img_feat_flat, obj_target_vec=obj_target_vec)
        return obj_choice, pair_choice


class SingleConcatCNN(SingleTranceNet):

    def __init__(
            self, *args, cnn_c_kernels=[16, 32, 64, 64],
            cnn_s_kernels=[5, 3, 3, 3], **kwargs):
        encoder = CNNEncoder(
            'concat', c_kernels=cnn_c_kernels, s_kernels=cnn_s_kernels)
        super().__init__(encoder, *args, **kwargs)


class SingleSubtractCNN(SingleTranceNet):

    def __init__(
            self, *args, cnn_c_kernels=[16, 32, 64, 64],
            cnn_s_kernels=[5, 3, 3, 3], **kwargs):
        encoder = CNNEncoder(
            'subtract', c_kernels=cnn_c_kernels, s_kernels=cnn_s_kernels)
        super().__init__(encoder, *args, **kwargs)


class SingleConcatResNet(SingleTranceNet):

    def __init__(self, *args, arch="resnet18", **kwargs):
        encoder = ResNetEncoder('concat', arch=arch)
        super().__init__(encoder, *args, **kwargs)


class SingleSubtractResNet(SingleTranceNet):

    def __init__(self, *args, arch="resnet18", **kwargs):
        encoder = ResNetEncoder('subtract', arch=arch)
        super().__init__(encoder, *args, **kwargs)


class SingleDUDA(SingleTranceNet):
    def __init__(self, *args, arch="resnet18", **kwargs):
        encoder = DUDAEncoder(arch)
        super().__init__(encoder, *args, **kwargs)


class SingleBCNN(nn.Module):

    def __init__(
            self, arch='vgg11_bn', height=240, width=320, c_obj=19, n_pair=33,
            fc_img_en=[256, 256], fc_fusion=[256, 256]):
        super().__init__()
        self.bcnn_encoder = BCNNEncoder(arch)
        bcnn_out = self.bcnn_encoder.get_output_shape(height, width)
        c_out = fc_img_en[0]
        self.fc = nn.Linear(bcnn_out, c_out)
        self.decoder = TransformationDecoder(
            c_out, c_obj, n_pair, fc_img_en, fc_fusion)

    def forward(self, init, fin, init_desc, obj_target_vec=None):
        img_feat = self.fc(self.bcnn_encoder(init, fin))
        obj_choice, pair_choice = self.decoder(
            init_desc, img_feat, obj_target_vec=obj_target_vec)
        return obj_choice, pair_choice
