import logging

import torch
from torch import nn

from .components.block import FCBlocks
from .components.decoder import TransformationDecoder
from .components.encoder import (
    BCNNEncoder,
    CNNEncoder,
    DUDAEncoder,
    ResNetEncoder,
)
from .components.x_transformer import Decoder

logger = logging.getLogger(__name__)


# ==============================================================================
# TranceNet Models for Single Transformation
# ==============================================================================


class SingleTranceNet(nn.Module):
    def __init__(
        self,
        encoder,
        height=120,
        width=160,
        c_obj=19,
        n_pair=33,
        fc_img_en=[1024, 256],
        fc_fusion=[256, 256],
    ):
        super().__init__()
        self.encoder = encoder
        c_out, h_out, w_out = self.encoder.get_output_shape(height, width)
        c_flat = c_out * h_out * w_out
        self.decoder = TransformationDecoder(
            c_flat, c_obj, n_pair, fc_img_en, fc_fusion
        )

    def forward(
        self,
        init: torch.Tensor = None,
        fin: torch.Tensor = None,
        init_desc: torch.Tensor = None,
        obj_target_vec: torch.Tensor = None,
        **kwargs,
    ):
        img_feat = self.encoder(init, fin)
        img_feat_flat = img_feat.flatten(start_dim=1)
        obj_choice, pair_choice = self.decoder(
            init_desc, img_feat_flat, obj_target_vec=obj_target_vec
        )

        return {
            "obj_choice": obj_choice,
            "pair_choice": pair_choice,
        }


class SingleConcatCNN(SingleTranceNet):
    def __init__(
        self,
        *args,
        cnn_c_kernels=[16, 32, 64, 64],
        cnn_s_kernels=[5, 3, 3, 3],
        **kwargs,
    ):
        encoder = CNNEncoder(
            "concat", c_kernels=cnn_c_kernels, s_kernels=cnn_s_kernels
        )
        super().__init__(encoder, *args, **kwargs)


class SingleSubtractCNN(SingleTranceNet):
    def __init__(
        self,
        *args,
        cnn_c_kernels=[16, 32, 64, 64],
        cnn_s_kernels=[5, 3, 3, 3],
        **kwargs,
    ):
        encoder = CNNEncoder(
            "subtract", c_kernels=cnn_c_kernels, s_kernels=cnn_s_kernels
        )
        super().__init__(encoder, *args, **kwargs)


class SingleConcatResNet(SingleTranceNet):
    def __init__(self, *args, arch="resnet18", **kwargs):
        encoder = ResNetEncoder("concat", arch=arch)
        super().__init__(encoder, *args, **kwargs)


class SingleSubtractResNet(SingleTranceNet):
    def __init__(self, *args, arch="resnet18", **kwargs):
        encoder = ResNetEncoder("subtract", arch=arch)
        super().__init__(encoder, *args, **kwargs)


class SingleDUDA(SingleTranceNet):
    def __init__(self, *args, arch="resnet18", **kwargs):
        encoder = DUDAEncoder(arch)
        super().__init__(encoder, *args, **kwargs)


class SingleBCNN(nn.Module):
    def __init__(
        self,
        arch="vgg11_bn",
        height=240,
        width=320,
        c_obj=19,
        n_pair=33,
        fc_img_en=[256, 256],
        fc_fusion=[256, 256],
    ):
        super().__init__()
        self.bcnn_encoder = BCNNEncoder(arch)
        bcnn_out = self.bcnn_encoder.get_output_shape(height, width)
        c_out = fc_img_en[0]
        self.fc = nn.Linear(bcnn_out, c_out)
        self.decoder = TransformationDecoder(
            c_out, c_obj, n_pair, fc_img_en, fc_fusion
        )

    def forward(
        self,
        init: torch.Tensor = None,
        fin: torch.Tensor = None,
        init_desc: torch.Tensor = None,
        obj_target_vec: torch.Tensor = None,
        **kwargs,
    ):
        img_feat = self.fc(self.bcnn_encoder(init, fin))
        obj_choice, pair_choice = self.decoder(
            init_desc, img_feat, obj_target_vec=obj_target_vec
        )
        return {
            "obj_choice": obj_choice,
            "pair_choice": pair_choice,
        }


# ==============================================================================
# TranceNet Models for Multiple Transformations
# ==============================================================================


class TranceNet(nn.Module):
    def __init__(
        self,
        encoder,
        height=240,
        width=320,
        c_obj=19,
        n_pair=33,
        rnn="GRU",
        teacher_forcing=True,
        c_pair_embed=8,
        max_step=4,
        c_hidden=[1024, 128],
        fc_img_en=[128, 128],
        fc_fusion=[128, 128],
        pretrained: str = None,
        state_dict_prefix: str = "model.",
        rnn_dropout=0,
        decoder_dropout=0,
    ):
        super().__init__()
        self.max_length = max_step + 1
        self.rnn = rnn
        self.teacher_forcing = teacher_forcing
        self.encoder = encoder
        fc_in = self.get_fc_in(height, width)
        self.fc = FCBlocks(fc_in, c_hidden)
        self.prepare_context_encoder(
            c_obj, n_pair, c_pair_embed, rnn, c_hidden, dropout=rnn_dropout
        )
        self.decoder = TransformationDecoder(
            c_hidden[-1],
            c_obj,
            n_pair + 1,
            fc_img_en,
            fc_fusion,
            dropout=decoder_dropout,
        )
        self.reset_parameters()

        if pretrained:
            self.load_pretrained(pretrained, prefix=state_dict_prefix)

    def load_pretrained(self, pretrained, prefix="model."):
        logger.info(f"Loading pretrained weights from {pretrained}")
        weights = torch.load(pretrained, map_location="cpu")["state_dict"]
        weights = {k.replace(prefix, ""): v for k, v in weights.items()}
        self.load_state_dict(weights)

    def get_fc_in(self, height, width):
        c_out, h_out, w_out = self.encoder.get_output_shape(height, width)
        c_flat = c_out * h_out * w_out
        return c_flat

    def prepare_context_encoder(
        self, c_obj, n_pair, c_pair_embed, rnn, c_hidden, dropout=None
    ):
        assert rnn in ["GRU", "LSTM"]

        self.no_obj = nn.Parameter(torch.zeros(c_obj))
        self.obj_init_input = nn.Parameter(torch.zeros(c_obj))
        self.embed_pair = nn.Embedding(n_pair + 3, c_pair_embed)
        self.pair_init_input = nn.Parameter(torch.zeros(c_pair_embed))

        c_in = c_obj + c_pair_embed
        RNN = getattr(nn, rnn)
        self.rnn_unit = RNN(
            c_in, c_hidden[-1], batch_first=True, dropout=dropout
        )

    def reset_parameters(self):
        nn.init.normal_(self.obj_init_input)
        nn.init.normal_(self.pair_init_input)
        nn.init.normal_(self.no_obj)

    def decode_choice(self, outputs, init_desc, obj_target_vec=None):
        B, L, H = outputs.shape
        outputs = outputs.reshape(-1, H)

        B, N, C = init_desc.shape
        init_desc = init_desc.unsqueeze(1).expand(-1, L, -1, -1)
        init_desc = init_desc.reshape(B * L, N, C)

        if type(obj_target_vec) is torch.Tensor:
            obj_target_vec = obj_target_vec.reshape(B * L, -1)

        obj_choice, obj_vec, pair_choice = self.decoder(
            init_desc,
            outputs,
            output_obj_vec=True,
            obj_target_vec=obj_target_vec,
        )

        obj_choice = obj_choice.reshape(B, L, -1)
        obj_vec = obj_vec.reshape(B, L, -1)
        pair_choice = pair_choice.reshape(B, L, -1)
        return obj_choice, obj_vec, pair_choice

    def compute_h0(self, init, fin):
        img_feat = self.encoder(init, fin)
        img_feat_flat = img_feat.flatten(start_dim=1)
        h0 = self.fc(img_feat_flat).unsqueeze(0)
        return h0

    def teacher_forcing_forward(
        self, x0_obj, x0_pair, h0, init_desc, obj_target_vec, pair_target
    ):
        x_obj = torch.cat([x0_obj, obj_target_vec[:, :-1, :]], dim=1)
        x_pair = torch.cat(
            [x0_pair, self.embed_pair(pair_target[:, :-1])], dim=1
        )
        x = torch.cat([x_obj, x_pair], dim=2)
        outputs, _ = self.rnn_unit(x, h0)
        obj_choices, _, pair_choices = self.decode_choice(
            outputs, init_desc, obj_target_vec
        )
        return obj_choices, pair_choices

    def iterative_forward(self, x0_obj, x0_pair, h0, init_desc):
        obj, pair = x0_obj, x0_pair
        hidden = h0
        obj_choices, pair_choices = [], []
        for i in range(self.max_length):
            x_in = torch.cat([obj, pair], dim=2)
            output, hidden = self.rnn_unit(x_in, hidden)
            obj_choice, obj, pair_choice = self.decode_choice(output, init_desc)
            obj_choices.append(obj_choice)
            pair_choices.append(pair_choice)
            pair = self.embed_pair(pair_choice.argmax(dim=2))
        obj_choices = torch.cat(obj_choices, dim=1)
        pair_choices = torch.cat(pair_choices, dim=1)
        return obj_choices, pair_choices

    def forward(
        self,
        init: torch.Tensor = None,
        fin: torch.Tensor = None,
        init_desc: torch.Tensor = None,
        obj_target_vec: torch.Tensor = None,
        pair_target: torch.Tensor = None,
        **kwargs,
    ):
        h0 = self.compute_h0(init, fin)
        if self.rnn == "LSTM":
            h0 = (h0, torch.zeros_like(h0))
        B = init.shape[0]
        init_desc = torch.cat([init_desc, self.no_obj.expand(B, 1, -1)], dim=1)
        x0_obj = self.obj_init_input.expand(B, 1, -1)
        x0_pair = self.pair_init_input.expand(B, 1, -1)
        if self.training and self.teacher_forcing:
            obj_choices, pair_choices = self.teacher_forcing_forward(
                x0_obj, x0_pair, h0, init_desc, obj_target_vec, pair_target
            )
        else:
            obj_choices, pair_choices = self.iterative_forward(
                x0_obj, x0_pair, h0, init_desc
            )

        return {
            "obj_choice": obj_choices,
            "pair_choice": pair_choices,
        }


class ConcatCNN(TranceNet):
    def __init__(
        self,
        *args,
        cnn_c_kernels=[16, 32, 64, 64],
        cnn_s_kernels=[5, 3, 3, 3],
        **kwargs,
    ):
        encoder = CNNEncoder(
            "concat", c_kernels=cnn_c_kernels, s_kernels=cnn_s_kernels
        )
        super().__init__(encoder, *args, **kwargs)


class SubtractCNN(TranceNet):
    def __init__(
        self,
        *args,
        cnn_c_kernels=[16, 32, 64, 64],
        cnn_s_kernels=[5, 3, 3, 3],
        **kwargs,
    ):
        encoder = CNNEncoder(
            "subtract", c_kernels=cnn_c_kernels, s_kernels=cnn_s_kernels
        )
        super().__init__(encoder, *args, **kwargs)


class ConcatResNet(TranceNet):
    def __init__(
        self, *args, arch="resnet18", fix_encoder_weights=False, **kwargs
    ):
        encoder = ResNetEncoder(
            "concat", arch=arch, fix_weights=fix_encoder_weights
        )
        super().__init__(encoder, *args, **kwargs)


class SubtractResNet(TranceNet):
    def __init__(
        self, *args, arch="resnet18", fix_encoder_weights=False, **kwargs
    ):
        encoder = ResNetEncoder(
            "subtract", arch=arch, fix_weights=fix_encoder_weights
        )
        super().__init__(encoder, *args, **kwargs)


class DUDA(TranceNet):
    def __init__(self, *args, arch="resnet18", **kwargs):
        encoder = DUDAEncoder(arch)
        super().__init__(encoder, *args, **kwargs)


class BCNN(TranceNet):
    def __init__(self, *args, arch="resnet18", **kwargs):
        encoder = BCNNEncoder(arch)
        super().__init__(encoder, *args, **kwargs)

    def get_fc_in(self, height, width):
        return self.encoder.get_output_shape(height, width)

    def compute_h0(self, init, fin):
        img_feat = self.encoder(init, fin)
        h0 = self.fc(img_feat).unsqueeze(0)
        return h0


class TranceFormer(nn.Module):
    def __init__(
        self,
        encoder,
        height=240,
        width=320,
        c_obj=19,
        n_pair=33,
        teacher_forcing=True,
        c_pair_embed=8,
        max_step=4,
        c_hidden=[128],
        fc_img_en=[128],
        fc_fusion=[128],
        pretrained: str = None,
        state_dict_prefix: str = "model.",
        context_encoder_dropout=0,
        decoder_dropout=0,
    ):
        super().__init__()
        self.max_length = max_step + 1
        self.teacher_forcing = teacher_forcing
        self.encoder = encoder
        fc_in = self.get_fc_in(height, width)
        self.fc = FCBlocks(fc_in, c_hidden)
        self.prepare_context_encoder(
            c_obj,
            n_pair,
            c_pair_embed,
            c_hidden,
            dropout=context_encoder_dropout,
        )
        self.decoder = TransformationDecoder(
            c_hidden[-1],
            c_obj,
            n_pair + 1,
            fc_img_en,
            fc_fusion,
            dropout=decoder_dropout,
        )
        self.reset_parameters()

        if pretrained:
            self.load_pretrained(pretrained, prefix=state_dict_prefix)

    def load_pretrained(self, pretrained, prefix="model."):
        logger.info(f"Loading pretrained weights from {pretrained}")
        weights = torch.load(pretrained, map_location="cpu")["state_dict"]
        weights = {k.replace(prefix, ""): v for k, v in weights.items()}
        self.load_state_dict(weights)

    def get_fc_in(self, height, width):
        c_out, h_out, w_out = self.encoder.get_output_shape(height, width)
        c_flat = c_out * h_out * w_out
        return c_flat

    def prepare_context_encoder(
        self, c_obj, n_pair, c_pair_embed, c_hidden, dropout=0.1
    ):
        self.no_obj = nn.Parameter(torch.zeros(c_obj))
        self.obj_init_input = nn.Parameter(torch.zeros(c_obj))
        self.embed_pair = nn.Embedding(n_pair + 3, c_pair_embed)
        self.pair_init_input = nn.Parameter(torch.zeros(c_pair_embed))

        c_in = c_obj + c_pair_embed

        self.linear_in = nn.Linear(c_in, c_hidden[-1])
        self.context_encoder = Decoder(
            dim=c_hidden[-1],
            depth=1,
            heads=8,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rel_pos_bias=True,
        )

    def reset_parameters(self):
        nn.init.normal_(self.obj_init_input)
        nn.init.normal_(self.pair_init_input)
        nn.init.normal_(self.no_obj)

    def decode_choice(self, outputs, init_desc, obj_target_vec=None):
        B, L, H = outputs.shape
        outputs = outputs.reshape(-1, H)

        B, N, C = init_desc.shape
        init_desc = init_desc.unsqueeze(1).expand(-1, L, -1, -1)
        init_desc = init_desc.reshape(B * L, N, C)

        if type(obj_target_vec) is torch.Tensor:
            obj_target_vec = obj_target_vec.reshape(B * L, -1)

        obj_choice, obj_vec, pair_choice = self.decoder(
            init_desc,
            outputs,
            output_obj_vec=True,
            obj_target_vec=obj_target_vec,
        )

        obj_choice = obj_choice.reshape(B, L, -1)
        obj_vec = obj_vec.reshape(B, L, -1)
        pair_choice = pair_choice.reshape(B, L, -1)
        return obj_choice, obj_vec, pair_choice

    def compute_h0(self, init, fin):
        img_feat = self.encoder(init, fin)
        img_feat_flat = img_feat.flatten(start_dim=1)
        h0 = self.fc(img_feat_flat).unsqueeze(1)
        return h0

    def teacher_forcing_forward(
        self, x0_obj, x0_pair, h0, init_desc, obj_target_vec, pair_target
    ):
        x_obj = torch.cat([x0_obj, obj_target_vec[:, :-1, :]], dim=1)
        x_pair = torch.cat(
            [x0_pair, self.embed_pair(pair_target[:, :-1])], dim=1
        )
        x = torch.cat([x_obj, x_pair], dim=2)
        x = self.linear_in(x)
        outputs = self.context_encoder(x + h0)
        obj_choices, _, pair_choices = self.decode_choice(
            outputs, init_desc, obj_target_vec
        )
        return obj_choices, pair_choices

    def iterative_forward(self, x0_obj, x0_pair, h0, init_desc):
        obj, pair = x0_obj, x0_pair
        hidden = h0
        obj_choices, pair_choices = [], []
        for i in range(self.max_length):
            x_in = torch.cat([obj, pair], dim=2)
            x_in = self.linear_in(x_in) + hidden
            if i == 0:
                x = x_in
            else:
                x = torch.cat([x, x_in], dim=1)
            output = self.context_encoder(x)
            obj_choice, obj, pair_choice = self.decode_choice(
                output[:, -1:, :], init_desc
            )
            obj_choices.append(obj_choice)
            pair_choices.append(pair_choice)
            pair = self.embed_pair(pair_choice.argmax(dim=2))
        obj_choices = torch.cat(obj_choices, dim=1)
        pair_choices = torch.cat(pair_choices, dim=1)
        return obj_choices, pair_choices

    def forward(
        self,
        init: torch.Tensor = None,
        fin: torch.Tensor = None,
        init_desc: torch.Tensor = None,
        obj_target_vec: torch.Tensor = None,
        pair_target: torch.Tensor = None,
        **kwargs,
    ):
        h0 = self.compute_h0(init, fin)
        B = init.shape[0]
        init_desc = torch.cat([init_desc, self.no_obj.expand(B, 1, -1)], dim=1)
        x0_obj = self.obj_init_input.expand(B, 1, -1)
        x0_pair = self.pair_init_input.expand(B, 1, -1)
        if self.training and self.teacher_forcing:
            obj_choices, pair_choices = self.teacher_forcing_forward(
                x0_obj, x0_pair, h0, init_desc, obj_target_vec, pair_target
            )
        else:
            obj_choices, pair_choices = self.iterative_forward(
                x0_obj, x0_pair, h0, init_desc
            )

        return {
            "obj_choice": obj_choices,
            "pair_choice": pair_choices,
        }


class ConcatResNetFormer(TranceFormer):
    def __init__(
        self, *args, arch="resnet18", fix_encoder_weights=False, **kwargs
    ):
        encoder = ResNetEncoder(
            "concat", arch=arch, fix_weights=fix_encoder_weights
        )
        super().__init__(encoder, *args, **kwargs)


class SubtractResNetFormer(TranceFormer):
    def __init__(
        self, *args, arch="resnet18", fix_encoder_weights=False, **kwargs
    ):
        encoder = ResNetEncoder(
            "subtract", arch=arch, fix_weights=fix_encoder_weights
        )
        super().__init__(encoder, *args, **kwargs)
