import logging
import math
from collections import namedtuple
from functools import partial
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

# from .autoregressive_wrapper import AutoregressiveWrapper

logger = logging.getLogger(__name__)

# constants

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple(
    "Intermediates", ["pre_softmax_attn", "post_softmax_attn"]
)

LayerIntermediates = namedtuple(
    "Intermediates", ["hiddens", "attn_intermediates"]
)

# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class not_equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x != self.val


class equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x == self.val


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")


# init helpers


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


# keyword argument helpers


def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d
    )
    kwargs_without_prefix = dict(
        map(
            lambda x: (x[0][len(prefix) :], x[1]),
            tuple(kwargs_with_prefix.items()),
        )
    )
    return kwargs_without_prefix, kwargs


# initializations


def deepnorm_init(
    transformer, beta, module_name_match_list=[".ff.", ".to_v", ".to_out"]
):
    for name, module in transformer.named_modules():
        if type(module) != nn.Linear:
            continue

        needs_beta_gain = any(
            map(lambda substr: substr in name, module_name_match_list)
        )
        gain = beta if needs_beta_gain else 1
        nn.init.xavier_normal_(module.weight.data, gain=gain)

        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)


# activations


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


# embedding


class TokenEmbedding(nn.Module):
    def __init__(self, dim, num_tokens, l2norm_embed=False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x)
        return l2norm(token_emb) if self.l2norm_embed else token_emb


# positional embeddings


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim**-0.5 if not l2norm_embed else 1.0
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb.weight, std=1e-5)

    def forward(self, x, pos=None):
        seq_len = x.shape[1]
        assert (
            seq_len <= self.max_seq_len
        ), f"you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}"

        if not exists(pos):
            pos = torch.arange(seq_len, device=x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, pos=None, seq_dim=1, offset=0):
        if not exists(pos):
            pos = torch.arange(x.shape[seq_dim], device=x.device)

        pos = pos.type_as(self.inv_freq) + offset
        sinusoid_inp = pos.unsqueeze(-1) * self.inv_freq
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb


class RelativePositionBias(nn.Module):
    def __init__(
        self, scale, causal=False, num_buckets=32, max_distance=128, heads=8
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> h i j")
        return qk_dots + (bias * self.scale)


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance=False, norm=False):
        super().__init__()
        assert (
            depth >= 1
        ), "depth for dynamic position bias MLP must be greater or equal to 1"
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(
            nn.Sequential(
                nn.Linear(1, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                nn.ReLU(),
            )
        )

        for _ in range(depth - 1):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim) if norm else nn.Identity(),
                    nn.ReLU(),
                )
            )

        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, qk_dots):
        n, device, dtype = qk_dots.shape[-1], qk_dots.device, qk_dots.dtype

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device=device)
        context_arange = torch.arange(n, device=device)
        indices = rearrange(seq_arange, "i -> i 1") - rearrange(
            context_arange, "j -> 1 j"
        )
        indices += n - 1

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device=device, dtype=dtype)
        pos = rearrange(pos, "... -> ... 1")

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(
                pos.abs() + 1
            )  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases
        bias = pos[indices]
        bias = rearrange(bias, "i j h -> h i j")
        return qk_dots + bias


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(
            rearrange(j_arange, "j -> 1 1 j")
            - rearrange(i_arange, "i -> 1 i 1")
        )
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer("bias", bias, persistent=False)

        return qk_dots + self.bias


class LearnedAlibiPositionalBias(AlibiPositionalBias):
    def __init__(self, heads):
        super().__init__(heads)
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        def get_slopes(param):
            return F.pad(param.exp(), (0, 0, 0, 0, 0, h - param.shape[0]))

        if exists(self.bias) and self.bias.shape[-1] >= j:
            bias = self.bias[..., :i, :j]
        else:
            bias = self.get_bias(i, j, device)
            self.register_buffer("bias", bias, persistent=False)

        slopes = get_slopes(self.learned_logslopes)
        bias = bias * slopes

        return qk_dots + bias


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


# norms


class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def _scale(self, x):
        return x * self.value

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        if not isinstance(out, tuple):
            return self._scale(out)

        return (self.scale_(out[0]), *out[1:])


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# residual and residual gates


class Residual(nn.Module):
    def __init__(self, dim, scale_residual=False, scale_residual_constant=1.0):
        super().__init__()
        self.residual_scale = (
            nn.Parameter(torch.ones(dim)) if scale_residual else None
        )
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual


class GRUGating(nn.Module):
    def __init__(self, dim, scale_residual=False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = (
            nn.Parameter(torch.ones(dim)) if scale_residual else None
        )

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, "b n d -> (b n) d"),
            rearrange(residual, "b n d -> (b n) d"),
        )

        return gated_output.reshape_as(x)


# token shifting


def shift(t, amount, mask=None):
    if amount == 0:
        return t
    else:
        amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.0)

    return F.pad(t, (0, 0, amount, -amount), value=0.0)


class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get("mask", None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(
            map(
                lambda args: shift(*args, mask=mask),
                zip(segments_to_shift, shifts),
            )
        )
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


# feedforward


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        swish=False,
        relu_squared=False,
        post_act_ln=False,
        dropout=0.0,
        no_bias=False,
        zero_init_output=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        project_in = (
            nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not no_bias), activation
            )
            if not glu
            else GLU(dim, inner_dim, activation)
        )

        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=not no_bias),
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)


# attention.


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=DEFAULT_DIM_HEAD,
        heads=8,
        causal=False,
        talking_heads=False,
        head_scale=False,
        sparse_topk=None,
        # use_entmax15=False,
        num_mem_kv=0,
        dropout=0.0,
        on_attn=False,
        gate_values=False,
        zero_init_output=False,
        max_attend_past=None,
        qk_norm=False,
        qk_norm_groups=1,
        qk_norm_scale=1,
        one_kv_head=False,
        shared_kv=False,
        value_dim_head=None,
    ):
        super().__init__()
        self.scale = dim_head**-0.5

        self.heads = heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        value_dim_head = default(value_dim_head, dim_head)
        q_dim = k_dim = dim_head * heads
        v_dim = out_dim = value_dim_head * heads

        self.one_kv_head = one_kv_head
        if one_kv_head:
            k_dim = dim_head
            v_dim = value_dim_head
            out_dim = v_dim * heads

        self.to_q = nn.Linear(dim, q_dim, bias=False)
        self.to_k = nn.Linear(dim, k_dim, bias=False)

        # shared key / values, for further memory savings during inference
        assert not (
            shared_kv and value_dim_head != dim_head
        ), "key and value head dimensions must be equal for shared key / values"
        self.to_v = nn.Linear(dim, v_dim, bias=False) if not shared_kv else None

        # dropout
        self.dropout = nn.Dropout(dropout)

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, out_dim)
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 1)

        # cosine sim attention
        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.qk_norm_scale = qk_norm_scale

        assert (not qk_norm) or (
            dim_head % qk_norm_groups
        ) == 0, "dimension per attention head must be divisible by the qk norm groups"
        assert not (
            qk_norm and (dim_head // qk_norm_groups) <= 2
        ), "the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)"

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(
                heads, heads, 1, bias=False
            )
            self.post_softmax_talking_heads = nn.Conv2d(
                heads, heads, 1, bias=False
            )

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        # self.attn_fn = (
        #     entmax15
        #     if use_entmax15
        #     else partial(F.softmax, dtype=torch.float32)
        # )
        self.attn_fn = partial(F.softmax, dtype=torch.float32)

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = (
            nn.Sequential(nn.Linear(out_dim, dim * 2, bias=False), nn.GLU())
            if on_attn
            else nn.Linear(out_dim, dim, bias=False)
        )

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        rel_pos=None,
        sinusoidal_emb=None,
        rotary_pos_emb=None,
        prev_attn=None,
        mem=None,
    ):
        b, n, _, h, talking_heads, head_scale, scale, device, has_context = (
            *x.shape,
            self.heads,
            self.talking_heads,
            self.head_scale,
            self.scale,
            x.device,
            exists(context),
        )
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input) if exists(self.to_v) else k

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        if not self.one_kv_head:
            k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (k, v)
            )

        if exists(rotary_pos_emb) and not has_context:
            sep = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(
                lambda t: (t[..., :sep], t[..., sep:]), (q, k, v)
            )
            ql, kl, vl = map(
                lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl)
            )
            q, k, v = map(
                lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr))
            )

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(
                mask, lambda: torch.ones((b, n), device=device).bool()
            )
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(
                k_mask,
                lambda: torch.ones((b, k.shape[-2]), device=device).bool(),
            )
            q_mask = rearrange(q_mask, "b i -> b 1 i 1")
            k_mask = rearrange(k_mask, "b j -> b 1 1 j")
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(
                lambda t: repeat(t, "h n d -> b h n d", b=b),
                (self.mem_k, self.mem_v),
            )
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups=self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

        kv_einsum_eq = "b h j d" if not self.one_kv_head else "b j d"

        dots = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots.clone()

        if talking_heads:
            dots = self.pre_softmax_talking_heads(dots)

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if exists(attn_mask):
            assert (
                2 <= attn_mask.ndim <= 4
            ), "attention mask must have greater than 2 dimensions but less than or equal to 4"
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, "i j -> 1 1 i j")
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, "h i j -> 1 h i j")
            dots.masked_fill_(~attn_mask, mask_value)

        if exists(self.max_attend_past):
            i, j = dots.shape[-2:]
            range_q = torch.arange(j - i, j, device=device)
            range_k = torch.arange(j, device=device)
            dist = rearrange(range_q, "i -> 1 1 i 1") - rearrange(
                range_k, "j -> 1 1 1 j"
            )
            mask = dist > self.max_attend_past
            dots.masked_fill_(mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, "i -> 1 1 i 1") < rearrange(r, "j -> 1 1 1 j")
            mask = F.pad(mask, (j - i, 0), value=False)
            dots.masked_fill_(mask, mask_value)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        if talking_heads:
            attn = self.post_softmax_talking_heads(attn)

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        if head_scale:
            out = out * self.head_scale_params

        out = rearrange(out, "b h n d -> b n (h d)")

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * gates.sigmoid()

        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn,
        )

        return self.to_out(out), intermediates


class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        alibi_pos_bias=False,
        alibi_num_heads=None,
        alibi_learned=False,
        rel_pos_bias=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        dynamic_pos_bias=False,
        dynamic_pos_bias_log_distance=False,
        dynamic_pos_bias_mlp_depth=2,
        dynamic_pos_bias_norm=False,
        position_infused_attn=False,
        rotary_pos_emb=False,
        rotary_emb_dim=None,
        custom_layers=None,
        sandwich_coef=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        gate_residual=False,
        scale_residual=False,
        scale_residual_constant=1.0,
        deepnorm=False,
        shift_tokens=0,
        sandwich_norm=False,
        zero_init_branch_output=False,
        **kwargs,
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim("attn_", kwargs)

        dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.has_pos_emb = (
            position_infused_attn or rel_pos_bias or rotary_pos_emb
        )
        self.pia_pos_emb = (
            FixedPositionalEmbedding(dim) if position_infused_attn else None
        )

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = (
            RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None
        )

        assert not (
            alibi_pos_bias and rel_pos_bias
        ), "you can only choose Alibi positional bias or T5 relative positional bias, not both"
        assert (
            rel_pos_num_buckets <= rel_pos_max_distance
        ), "number of relative position buckets must be less than the relative position max distance"

        # relative positional bias

        self.rel_pos = None
        if rel_pos_bias:
            self.rel_pos = RelativePositionBias(
                scale=dim_head**0.5,
                causal=causal,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
            )
        elif dynamic_pos_bias:
            self.rel_pos = DynamicPositionBias(
                dim=dim // 4,
                heads=heads,
                log_distance=dynamic_pos_bias_log_distance,
                depth=dynamic_pos_bias_mlp_depth,
                norm=dynamic_pos_bias_norm,
            )
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert (
                alibi_num_heads <= heads
            ), "number of ALiBi heads must be less than the total number of heads"
            alibi_pos_klass = (
                LearnedAlibiPositionalBias
                if alibi_learned
                else AlibiPositionalBias
            )
            self.rel_pos = alibi_pos_klass(heads=alibi_num_heads)

        # determine deepnorm and residual scale

        if deepnorm:
            assert (
                scale_residual_constant == 1
            ), "scale residual constant is being overridden by deep norm settings"
            pre_norm = sandwich_norm = False
            scale_residual = True
            scale_residual_constant = (2 * depth) ** 0.25

        assert not (
            not pre_norm and sandwich_norm
        ), "sandwich norm cannot be used when not using prenorm"
        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross:
            default_block = ("a", "c", "f")
        elif cross_attend and only_cross:
            default_block = ("c", "f")
        else:
            default_block = ("a", "f")

        if macaron:
            default_block = ("f",) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, "zero_init_output": True}
            ff_kwargs = {**ff_kwargs, "zero_init_output": True}

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = (
                par_depth * 2 // 3
            )  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert (
                len(default_block) <= par_width
            ), "default block is too large for par_ratio"
            par_block = default_block + ("f",) * (
                par_width - len(default_block)
            )
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert (
                sandwich_coef > 0 and sandwich_coef <= depth
            ), "sandwich coefficient should be less than the depth"
            layer_types = (
                ("a",) * sandwich_coef
                + default_block * (depth - sandwich_coef)
                + ("f",) * sandwich_coef
            )
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(
            zip(self.layer_types, shift_tokens)
        ):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == "a":
                layer = Attention(
                    dim, heads=heads, causal=causal, **attn_kwargs
                )
            elif layer_type == "c":
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"invalid layer type {layer_type}")

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(
                    range(shift_range_lower, shift_range_upper), layer
                )

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(
                dim,
                scale_residual=scale_residual,
                scale_residual_constant=scale_residual_constant,
            )

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = (
                norm_fn() if not pre_norm and not is_last_layer else None
            )

            norms = nn.ModuleList(
                [pre_branch_norm, post_branch_norm, post_main_norm]
            )

            self.layers.append(nn.ModuleList([norms, layer, residual]))

        if deepnorm:
            init_gain = (8 * depth) ** -0.25
            deepnorm_init(self, init_gain)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        mems=None,
        return_hiddens=False,
    ):
        assert not (
            self.cross_attend ^ exists(context)
        ), "context must be passed in if cross_attend is set to True"

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(
                list(
                    map(
                        lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1],
                        mems,
                    )
                )
            )
            rotary_pos_emb = self.rotary_pos_emb(
                max_rotary_emb_length, x.device
            )

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(
            zip(self.layer_types, self.layers)
        ):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == "a":
                if return_hiddens:
                    hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            residual = x

            pre_branch_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_branch_norm):
                x = pre_branch_norm(x)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    attn_mask=attn_mask,
                    sinusoidal_emb=self.pia_pos_emb,
                    rel_pos=self.rel_pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    mem=layer_mem,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                )
            elif layer_type == "f":
                out = block(x)

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, residual)

            if layer_type in ("a", "c") and return_hiddens:
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens, attn_intermediates=intermediates
            )

            return x, intermediates

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on encoder"
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on decoder"
        super().__init__(causal=True, **kwargs)


class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True, only_cross=True, **kwargs)


class ViTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers,
        channels=3,
        num_classes=None,
        dropout=0.0,
        post_emb_norm=False,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, Encoder
        ), "attention layers must be an Encoder"
        assert (
            image_size % patch_size == 0
        ), "image dimensions must be divisible by the patch size"
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.post_emb_norm = (
            nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = (
            nn.Linear(dim, num_classes)
            if exists(num_classes)
            else nn.Identity()
        )

    def forward(self, img, return_embeddings=False):
        p = self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        n = x.shape[1]

        x = x + self.pos_embedding[:, :n]

        x = self.post_emb_norm(x)
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        if not exists(self.mlp_head) or return_embeddings:
            return x

        x = x.mean(dim=-2)
        return self.mlp_head(x)


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_mem_len=0.0,
        shift_mem_down=0,
        emb_dropout=0.0,
        post_emb_norm=False,
        num_memory_tokens=None,
        tie_embedding=False,
        use_fixed_pos_emb=False,
        use_abs_pos_emb=True,
        l2norm_embed=False,
        emb_frac_gradient=1.0,  # GLM-130B and Cogview successfully used this, set at 0.1
    ):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed
        self.token_emb = TokenEmbedding(
            emb_dim, num_tokens, l2norm_embed=l2norm_embed
        )

        if not attn_layers.has_pos_emb:
            if use_fixed_pos_emb:
                self.pos_emb = FixedPositionalEmbedding(emb_dim)
            elif use_abs_pos_emb:
                self.pos_emb = AbsolutePositionalEmbedding(
                    emb_dim, max_seq_len, l2norm_embed=l2norm_embed
                )
            else:
                self.pos_emb = always(0)
                logger.warning(
                    "using transformer without any positional embedding"
                )
        else:
            self.pos_emb = always(0)

        self.emb_frac_gradient = emb_frac_gradient  # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.post_emb_norm = (
            nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = (
            nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        )
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = (
            nn.Linear(dim, num_tokens)
            if not tie_embedding
            else lambda t: t @ self.token_emb.weight.t()
        )

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(num_memory_tokens, dim)
            )

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb.emb.weight, std=1e-5)
            return
        nn.init.kaiming_normal_(self.token_emb.emb.weight)

    def forward(
        self,
        x,
        return_embeddings=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        pos=None,
        **kwargs,
    ):
        b, n, device, num_mem, emb_frac_gradient = (
            *x.shape,
            x.device,
            self.num_memory_tokens,
            self.emb_frac_gradient,
        )
        return_hiddens = return_mems | return_attn

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos=pos) if not external_pos_emb else pos
        x = self.token_emb(x) + pos_emb

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
            x = torch.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = (
                mems[: self.shift_mem_down],
                mems[self.shift_mem_down :],
            )
            mems = [*mems_r, *mems_l]

        if return_hiddens:
            x, intermediates = self.attn_layers(
                x, mask=mask, mems=mems, return_hiddens=True, **kwargs
            )
        else:
            x = self.attn_layers(x, mask=mask, mems=mems, **kwargs)
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(
                    map(
                        lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens)
                    )
                )
                if exists(mems)
                else hiddens
            )
            new_mems = list(
                map(lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems)
            )
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(
                    lambda t: t.post_softmax_attn,
                    intermediates.attn_intermediates,
                )
            )
            return out, attn_maps

        return out


class ContinuousTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers,
        dim_in=None,
        dim_out=None,
        emb_dim=None,
        post_emb_norm=False,
        emb_dropout=0.0,
        use_abs_pos_emb=True,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.pos_emb = (
            AbsolutePositionalEmbedding(dim, max_seq_len)
            if (use_abs_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )

        self.post_emb_norm = (
            nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_in = (
            nn.Linear(dim_in, dim) if exists(dim_in) else nn.Identity()
        )

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.project_out = (
            nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()
        )

    def forward(
        self,
        x,
        return_embeddings=False,
        mask=None,
        return_attn=False,
        mems=None,
        pos=None,
        **kwargs,
    ):
        b, n, _, device = *x.shape, x.device

        x = self.project_in(x)
        x = x + self.pos_emb(x, pos=pos)

        x = self.post_emb_norm(x)
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, return_hiddens=True, **kwargs
        )
        x = self.norm(x)

        out = self.project_out(x) if not return_embeddings else x

        if return_attn:
            attn_maps = list(
                map(
                    lambda t: t.post_softmax_attn,
                    intermediates.attn_intermediates,
                )
            )
            return out, attn_maps

        return out
