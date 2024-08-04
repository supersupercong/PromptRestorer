import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY

# from .network_swinir_query import RSTB as RSTB_query
# from .network_swinir_kv import RSTB as RSTB_kv

from .network_swinir_key import RSTB as RSTB_key
from .network_swinir_value import RSTB as RSTB_value
from .network_swinir_query import RSTB as RSTB_query
from .network_swinir_kv import RSTB as RSTB_kv
from .network_swinir_qkv import RSTB as RSTB_qkv
from .fema_utils import ResBlock, CombineQuantBlock
from .vgg_arch import VGGFeatureExtractor
from timm.models.layers import DropPath
from .restormer import TransformerBlock
from .restormer import TransformerBlock_Query
from .restormer import TransformerBlock_Key_Value
from .restormer import TransformerBlock_QKV

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


## AdaMean
def adaptive_mean_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size))
    return normalized_feat + style_mean.expand(size)


## AdaStd
def adaptive_std_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat) / content_std.expand(size)
    return normalized_feat * style_std.expand(size)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)

    def dist(self, x, y):
        return torch.sum(x ** 2, dim=1, keepdim=True) + \
               torch.sum(y ** 2, dim=1) - 2 * \
               torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            codebook_loss = self.beta * ((z_q_gt.detach() - z) ** 2).mean()
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """

    def __init__(self, channels, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'none':
            self.norm = lambda x: x * 1.0
        else:
            assert 1 == 0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ResBlock_TransformerBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """

    def __init__(self, dim=32, num_heads=8, unit_num=3):
        super(ResBlock_TransformerBlock, self).__init__()
        self.channel = dim

        self.unit_num = unit_num

        self.TransformerBlock = nn.ModuleList()
        self.perception_kernel = nn.ModuleList()

        for i in range(self.unit_num):
            self.TransformerBlock.append(TransformerBlock(dim=self.channel, num_heads=num_heads))
            self.perception_kernel.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 1, 1), nn.Sigmoid()))

    def forward(self, input, perception_feature):
        tmp = input
        if perception_feature is not None:
            for i in range(self.unit_num):
                tmp = self.perception_kernel[i](perception_feature) * tmp + tmp
                tmp = self.TransformerBlock[i](tmp)
        else:
            for i in range(self.unit_num):
                tmp = self.TransformerBlock[i](tmp)

        out = 0.2 * tmp + input
        return out


class Refinement_TransformerBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """

    def __init__(self, dim=32, num_heads=2, unit_num=4):
        super(Refinement_TransformerBlock, self).__init__()
        self.channel = dim
        self.unit_num = unit_num
        self.TransformerBlock = nn.ModuleList()
        for i in range(self.unit_num):
            self.TransformerBlock.append(TransformerBlock(dim=self.channel, num_heads=num_heads))

    def forward(self, input):
        tmp = input
        for i in range(self.unit_num):
            tmp = self.TransformerBlock[i](tmp)

        out = 0.2 * tmp + input
        return out


class DenseResidualBlock(nn.Module):
    def __init__(self, unit_num, channel, num_heads=1):
        super(DenseResidualBlock, self).__init__()
        self.unit_num = unit_num
        self.channel = channel
        self.units = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        self.skip_connection = nn.ModuleList()
        self.conv = nn.Conv2d(self.channel, self.channel, 3, 1, 1)
        for i in range(self.unit_num):
            self.units.append(ResBlock_TransformerBlock(dim=self.channel, num_heads=num_heads))
            # self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel, self.channel, 1, 1)))
            # self.skip_connection.append(nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1)))

    def forward(self, x, perception_feature=None, encoder_feature=None):
        cat = []
        cat.append(x)
        tmp = x
        feature = []
        for i in range(self.unit_num):
            if encoder_feature is not None:
                if perception_feature is not None:
                    tmp = self.units[i](tmp, perception_feature)
                else:
                    tmp = self.units[i](tmp)
            else:
                if perception_feature is not None:
                    tmp = self.units[i](tmp, perception_feature)
                else:
                    tmp = self.units[i](tmp)
            feature.append(tmp)
            cat.append(tmp)
        return tmp, feature


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=192,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 **swin_opts,
                 ):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channel, channel_query_dict[input_res], 3, padding=1),
            ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
            ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
        )

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        # if LQ_stage:
        #     self.blocks.append(SwinLayers(**swin_opts))
        #     upsampler = nn.ModuleList()
        #     for i in range(2):
        #         in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
        #         upsampler.append(nn.Sequential(
        #             nn.Upsample(scale_factor=2),
        #             nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
        #             ResBlock(out_channel, out_channel, norm_type, act_type),
        #             ResBlock(out_channel, out_channel, norm_type, act_type),
        #         )
        #         )
        #         res = res * 2
        #
        #     self.blocks += upsampler

        self.LQ_stage = LQ_stage

    def forward(self, input):
        outputs = []
        x = self.in_conv(input)

        outputs.append(x)

        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)

        return outputs


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            nn.Upsample(scale_factor=2),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


class Mlp(nn.Module):
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


class Local_Perception(nn.Module):
    def __init__(self, dim, dim_pre, pooling_r=4, local_degregation_aware_restore_aware=True,
                 local_degregation_aware=True,
                 local_restore_aware=True, bias=True):
        super(Local_Perception, self).__init__()
        self.local_degregation_aware_restore_aware = local_degregation_aware_restore_aware
        self.local_degregation_aware = local_degregation_aware
        self.local_restore_aware = local_restore_aware

        if self.local_degregation_aware_restore_aware is True:
            self.degradation = nn.Sequential(
                nn.Conv2d(dim_pre, dim, 1, 1),
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))
            ###############################
            self.input = nn.Sequential(
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            )

            self.main_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )

            self.degradation_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )

            self.fusion1 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, bias=bias)
            self.ffn = nn.Sequential(
                # nn.Conv2d(dim, dim, kernel_size=1, stride=1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            )
        elif self.local_degregation_aware is True:
            self.degradation = nn.Sequential(
                nn.Conv2d(dim_pre, dim, 1, 1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))
            ###############################
            self.input = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            )

            self.degradation_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )
            self.ffn = nn.Sequential(
                # nn.Conv2d(dim, dim, kernel_size=1, stride=1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            )
        elif self.local_restore_aware is True:
            self.degradation = nn.Sequential(
                nn.Conv2d(dim_pre, dim, 1, 1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))
            ###############################
            self.input = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            )

            self.main_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )
            self.ffn = nn.Sequential(
                # nn.Conv2d(dim, dim, kernel_size=1, stride=1), nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            )
        self.layernorm1 = LayerNorm(dim_pre)
        self.layernorm2 = LayerNorm(dim)

    def forward(self, x, feature_perception=None):
        b, c, h, w = x.shape
        ############################### Soft_Concert ###############################
        if feature_perception is not None:
            if self.local_degregation_aware_restore_aware is True:
                degradation = self.degradation(self.layernorm1(feature_perception))
                degradation1 = degradation[:, c:, :, :]
                degradation2 = degradation[:, :c, :, :]
                input = self.input(self.layernorm2(x))
                input1 = input[:, c:, :, :]
                input2 = input[:, :c, :, :]
                ###############################
                main_kernel = self.main_kernel(torch.cat([input1, degradation1], dim=1))
                main_kernel = F.sigmoid(main_kernel)
                main_kernel_mul = torch.mul(main_kernel, degradation1)
                ###############################
                degradation_kernel = self.degradation_kernel(torch.cat([input2, degradation2], dim=1))
                degradation_kernel = F.sigmoid(degradation_kernel)
                degradation_kernel_mul = torch.mul(degradation_kernel, input2)
                out = self.fusion1(torch.cat([degradation_kernel_mul, main_kernel_mul], dim=1)) + x
                # out = degradation_kernel_mul + main_kernel_mul
                fusion1 = self.ffn(out) + out
                return fusion1
            # self.local_degregation_aware = local_degregation_aware
            # self.local_restore_aware = local_restore_aware

            elif self.local_degregation_aware is True:
                degradation = self.degradation(self.layernorm1(feature_perception))
                input = self.input(self.layernorm2(x))
                ###############################
                degradation_kernel = self.degradation_kernel(torch.cat([input, degradation], dim=1))
                degradation_kernel = F.sigmoid(degradation_kernel)
                out = torch.mul(degradation_kernel, input) + x
                fusion1 = self.ffn(out) + out
                return fusion1
            elif self.local_restore_aware is True:
                degradation = self.degradation(self.layernorm1(feature_perception))
                input = self.input(self.layernorm2(x))
                main_kernel = self.main_kernel(torch.cat([input, degradation], dim=1))
                main_kernel = F.sigmoid(main_kernel)
                out = torch.mul(main_kernel, degradation) + x
                fusion1 = self.ffn(out) + out
                return fusion1
        else:
            return x


class Global_Perception(nn.Module):
    def __init__(self, dim, dim_pre, num_heads=8, depth=2, res=(128, 128), pooling_r=4,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=True,
                 global_restore_aware=True, bias=True,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        # self.transformation = Perception_transformation(dim=dim, pooling_r=pooling_r, soft_perception=soft_perception,
        #                                              hard_perception=hard_perception, bias=bias)

        #######################################################################################################
        self.global_degregation_aware_restore_aware = global_degregation_aware_restore_aware
        self.global_degregation_aware = global_degregation_aware
        self.global_restore_aware = global_restore_aware

        if self.global_degregation_aware_restore_aware is True:
            self.Attention_qkv = TransformerBlock_QKV(dim, num_heads=num_heads)
            self.layernorm = LayerNorm(dim_pre)
            self.qkv_dwconv = nn.Sequential(nn.Conv2d(dim_pre, dim * 3, kernel_size=1, stride=1),
                                            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim,
                                                      bias=True))
        elif self.global_degregation_aware is True:
            self.Attention_q = TransformerBlock_Query(dim, num_heads=num_heads)
            self.layernorm = LayerNorm(dim_pre)
            self.q_dwconv = nn.Sequential(nn.Conv2d(dim_pre, dim * 2, kernel_size=1, stride=1),
                                          nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim,
                                                    bias=True))
        elif self.global_restore_aware is True:
            self.Attention_kv = TransformerBlock_Key_Value(dim, num_heads=num_heads)
            self.layernorm = LayerNorm(dim_pre)
            self.qkv_dwconv = nn.Sequential(nn.Conv2d(dim_pre, dim * 2, kernel_size=1, stride=1),
                                            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim,
                                                      bias=True))

    def forward(self, x, feature_perception=None):
        b, c, h, w = x.size()
        if feature_perception is not None:
            if self.global_degregation_aware_restore_aware is True:
                qkv = self.qkv_dwconv(self.layernorm(feature_perception))
                q_dwconv, k_dwconv, v_dwconv = qkv.chunk(3, dim=1)
                Attention_qkv = self.Attention_qkv(x, feature1=q_dwconv, feature2=k_dwconv,
                                                   feature3=v_dwconv)
                return Attention_qkv
            elif self.global_degregation_aware is True:
                q = self.q_dwconv(self.layernorm(feature_perception))
                Attention_q = self.Attention_q(x, feature=q)
                return Attention_q
            elif self.global_restore_aware is True:
                kv = self.qkv_dwconv(self.layernorm(feature_perception))
                k_dwconv, v_dwconv = kv.chunk(2, dim=1)
                Attention_kv = self.Attention_kv(x, feature1=k_dwconv, feature2=v_dwconv)
                return Attention_kv
        else:
            return x


class Basic_layer(nn.Module):
    def __init__(self, dim, dim_pre, num_heads=8, res=(128, 128), mlp_ratio=4., pooling_r=4, bias=True,
                 local_perception=True,
                 global_perception=True,
                 local_degregation_aware_restore_aware=True,
                 local_degregation_aware=True,
                 local_restore_aware=True,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=True,
                 global_restore_aware=True, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self._with_local_perception = local_perception
        self._with_global_perception = global_perception
        if self._with_local_perception is True:
            self.local_perception = Local_Perception(dim, dim_pre=dim_pre, pooling_r=pooling_r,
                                                     local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                                                     local_degregation_aware=local_degregation_aware,
                                                     local_restore_aware=local_restore_aware, bias=bias)

        if self._with_global_perception is True:
            self.global_perception = Global_Perception(dim=dim, dim_pre=dim_pre, num_heads=num_heads, res=res,
                                                       pooling_r=pooling_r,
                                                       global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                                                       global_degregation_aware=global_degregation_aware,
                                                       global_restore_aware=global_restore_aware, bias=bias)

        self.fusion = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=True)

    def forward(self, x, feature=None):
        B, C, H, W = x.size()

        if self._with_local_perception is True and self._with_global_perception is True:
            global_perception = self.global_perception(x, feature)
            local_perception = self.local_perception(x, feature)
            out = self.fusion(torch.cat([local_perception, global_perception], dim=1)) + x
            return out
        elif self._with_local_perception is True and self._with_global_perception is False:
            local_perception = self.local_perception(x, feature) + x
            out = local_perception
            return out
        elif self._with_local_perception is False and self._with_global_perception is True:
            global_perception = self.global_perception(x, feature) + x
            out = global_perception
            return out


class SFTLayer_torch(nn.Module):
    def __init__(self, dim, dim_pre):
        super(SFTLayer_torch, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(dim_pre, dim, 1)
        self.SFT_scale_conv1 = nn.Conv2d(dim, dim, 1)
        self.SFT_shift_conv0 = nn.Conv2d(dim_pre, dim, 1)
        self.SFT_shift_conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x0, feature):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(feature), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(feature), 0.01, inplace=True))
        return x0 * scale + shift


class _block(nn.Module):
    def __init__(self, dim, dim_pre, num_heads, length, res=(128, 128),
                 degregation_propagate=True,
                 local_perception=True,
                 global_perception=True,
                 local_degregation_aware_restore_aware=True,
                 local_degregation_aware=True,
                 local_restore_aware=True,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=True,
                 global_restore_aware=True,
                 perception_SFT=True, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.block_list = nn.ModuleList()
        self.length = length
        self.degregation_propagate = degregation_propagate

        if perception_SFT is False:
            self.basic_layer = Basic_layer(dim, dim_pre=dim_pre, num_heads=num_heads, res=res,
                                           local_perception=local_perception,
                                           global_perception=global_perception,
                                           local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                                           local_degregation_aware=local_degregation_aware,
                                           local_restore_aware=local_restore_aware,
                                           global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                                           global_degregation_aware=global_degregation_aware,
                                           global_restore_aware=global_restore_aware)
        else:
            self.basic_layer = SFTLayer_torch(dim=dim, dim_pre=dim_pre)
        self.drb1 = DenseResidualBlock(unit_num=length, channel=dim, num_heads=num_heads)

        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, encoder_feature=None, feature=None):
        ori = x
        b, c, h, w = x.size()
        if feature is not None:
            basic_layer = self.basic_layer(x, feature=feature)
            # basic_layer = adaptive_mean_normalization(content_feat=basic_layer, style_feat=ori)
        else:
            basic_layer = x

        if self.degregation_propagate is True:
            drb1, feature1_ = self.drb1(basic_layer, perception_feature=basic_layer, encoder_feature=encoder_feature)
        else:
            drb1, feature1_ = self.drb1(basic_layer, perception_feature=None, encoder_feature=encoder_feature)

        # conv = self.conv(drb1)

        # out = x + conv

        return drb1, feature1_


class Block(nn.Module):
    def __init__(self, dim, dim_pre, num_heads, length, res=(128, 128),
                 degregation_propagate=True,
                 local_perception=True,
                 global_perception=True,
                 local_degregation_aware_restore_aware=True,
                 local_degregation_aware=True,
                 local_restore_aware=True,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=True,
                 global_restore_aware=True,
                 perception_SFT=True, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.length = length

        self._block = _block(dim=dim, dim_pre=dim_pre, num_heads=num_heads, length=length, res=res,
                             degregation_propagate=degregation_propagate,
                             local_perception=local_perception,
                             global_perception=global_perception,
                             local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                             local_degregation_aware=local_degregation_aware,
                             local_restore_aware=local_restore_aware,
                             global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                             global_degregation_aware=global_degregation_aware,
                             global_restore_aware=global_restore_aware,
                             perception_SFT=perception_SFT)

    def forward(self, x, encoder_feature=None, feature=None):
        ori = x
        b, c, h, w = x.size()
        out, feature1_ = self._block(x, encoder_feature=encoder_feature, feature=feature)
        return out, feature1_


class Net_encoder_decoder(nn.Module):
    def __init__(self, channel_query_dict, channel_query_dict_pre,
                 length_list,
                 degregation_propagate=True,
                 local_perception=True,
                 global_perception=True,
                 local_degregation_aware_restore_aware=True,
                 local_degregation_aware=True,
                 local_restore_aware=True,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=True,
                 global_restore_aware=True,
                 perception_SFT=True):
        super().__init__()
        self.local_degregation_aware = local_degregation_aware
        self.channel_query_dict = channel_query_dict
        self.length_list = length_list
        self.enter = nn.Sequential(nn.Conv2d(3, channel_query_dict[256], 3, 1, 1))
        self.en_block1 = Block(channel_query_dict[256], channel_query_dict_pre[256], 2, length_list[0], res=(128, 128),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)

        self.c_128to64 = nn.Sequential(nn.Conv2d(channel_query_dict[256], channel_query_dict[256] // 2, kernel_size=3,
                                                 stride=1, padding=1, bias=True), nn.PixelUnshuffle(2))
        self.en_block2 = Block(channel_query_dict[128], channel_query_dict_pre[128], 4, length_list[1], res=(64, 64),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)
        self.c_64to32 = nn.Sequential(nn.Conv2d(channel_query_dict[128], channel_query_dict[128] // 2, kernel_size=3,
                                                stride=1, padding=1, bias=True), nn.PixelUnshuffle(2))
        self.bottom_block3 = Block(channel_query_dict[64], channel_query_dict_pre[64], 8, length_list[2], res=(16, 16),
                                   degregation_propagate=degregation_propagate,
                                   local_perception=local_perception,
                                   global_perception=global_perception,
                                   local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                                   local_degregation_aware=local_degregation_aware,
                                   local_restore_aware=local_restore_aware,
                                   global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                                   global_degregation_aware=global_degregation_aware,
                                   global_restore_aware=global_restore_aware,
                                   perception_SFT=perception_SFT)

        self.c_32to64 = nn.Sequential(nn.Conv2d(channel_query_dict[64], channel_query_dict[64] * 2, kernel_size=3,
                                                stride=1, padding=1, bias=True), nn.PixelShuffle(2))

        self.de_block2 = Block(channel_query_dict[128], channel_query_dict_pre[128], 4, length_list[1], res=(64, 64),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)

        self.c_64to128 = nn.Sequential(nn.Conv2d(channel_query_dict[128], channel_query_dict[128] * 2, kernel_size=3,
                                                 stride=1, padding=1, bias=True), nn.PixelShuffle(2))
        self.de_block3 = Block(channel_query_dict[256], channel_query_dict_pre[256], 2, length_list[0], res=(128, 128),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)

        self.exit = nn.Sequential(Refinement_TransformerBlock(channel_query_dict[256]),
                                  nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1))

        self.h = 32
        self.w = 32
        self.dim_embd = 256

    def _get_pos_embed(self, H, W):
        return F.interpolate(
            self.pos_embed.reshape(1, self.h, self.w, -1).permute(0, 3, 1, 2),
            size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward(self, x, feature):
        ori = x
        features = []
        enter = self.enter(x)
        en_block1, en_block1_feature11 = self.en_block1(enter, feature=feature[0])
        features.append(en_block1)

        c_128to64 = self.c_128to64(en_block1)
        en_block2, en_block2_feature11 = self.en_block2(c_128to64, feature=feature[1])
        features.append(en_block2)

        c_64to32 = self.c_64to32(en_block2)
        bottom_block3, bottom_block3_feature11 = self.bottom_block3(c_64to32, feature=feature[2])
        features.append(bottom_block3)
        c_32to64 = self.c_32to64(bottom_block3)
        de_block2, de_block2_feature11 = self.de_block2(c_32to64, encoder_feature=en_block2_feature11,
                                                        feature=feature[1])
        features.append(de_block2)
        c_64to128 = self.c_64to128(de_block2)
        de_block3, de_block3_feature11 = self.de_block3(c_64to128, encoder_feature=en_block1_feature11,
                                                        feature=feature[0])
        features.append(de_block3)
        exit = self.exit(de_block3)

        return exit + ori


class Learnable_degradation(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=128,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 ):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channel, channel_query_dict[input_res], 3, padding=1),
            ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
            ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
        )

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

    def forward(self, input):
        outputs = []
        x = self.in_conv(input)
        outputs.append(x)
        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)

        return outputs


class Encoder_decoder_learnable(nn.Module):
    def __init__(self, channel_query_dict, channel_query_dict_pre,
                 length_list, in_channel, encode_depth, gt_res, norm_type, act_type,
                 degregation_propagate=True,
                 local_perception=True,
                 global_perception=True,
                 local_degregation_aware_restore_aware=True,
                 local_degregation_aware=True,
                 local_restore_aware=True,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=True,
                 global_restore_aware=True,
                 perception_SFT=True):
        super().__init__()

        self.learnable_degration = Learnable_degradation(
            in_channel=in_channel,
            max_depth=encode_depth,
            input_res=gt_res,
            channel_query_dict=channel_query_dict,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.enter = nn.Sequential(nn.Conv2d(3, channel_query_dict[192], 3, 1, 1))
        self.en_block1 = Block(channel_query_dict[192], channel_query_dict_pre[192], 2, length_list[0], res=(128, 128),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)

        self.c_128to64 = nn.Sequential(nn.Conv2d(channel_query_dict[192], channel_query_dict[192] // 2, kernel_size=3,
                                                 stride=1, padding=1, bias=True), nn.PixelUnshuffle(2))
        self.en_block2 = Block(channel_query_dict[96], channel_query_dict_pre[96], 4, length_list[1], res=(64, 64),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)
        self.c_64to32 = nn.Sequential(nn.Conv2d(channel_query_dict[96], channel_query_dict[96] // 2, kernel_size=3,
                                                stride=1, padding=1, bias=True), nn.PixelUnshuffle(2))
        self.bottom_block3 = Block(channel_query_dict[48], channel_query_dict_pre[48], 8, length_list[2], res=(16, 16),
                                   degregation_propagate=degregation_propagate,
                                   local_perception=local_perception,
                                   global_perception=global_perception,
                                   local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                                   local_degregation_aware=local_degregation_aware,
                                   local_restore_aware=local_restore_aware,
                                   global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                                   global_degregation_aware=global_degregation_aware,
                                   global_restore_aware=global_restore_aware,
                                   perception_SFT=perception_SFT)

        self.c_32to64 = nn.Sequential(nn.Conv2d(channel_query_dict[48], channel_query_dict[48] * 2, kernel_size=3,
                                                stride=1, padding=1, bias=True), nn.PixelShuffle(2))

        self.de_block2 = Block(channel_query_dict[96], channel_query_dict_pre[96], 4, length_list[1], res=(64, 64),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)

        self.c_64to128 = nn.Sequential(nn.Conv2d(channel_query_dict[96], channel_query_dict[96] * 2, kernel_size=3,
                                                 stride=1, padding=1, bias=True), nn.PixelShuffle(2))
        self.de_block3 = Block(channel_query_dict[192], channel_query_dict_pre[192], 2, length_list[0], res=(128, 128),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)

        self.exit = nn.Sequential(Refinement_TransformerBlock(channel_query_dict[192]),
                                  nn.Conv2d(channel_query_dict[192], 3, 3, 1, 1))

        self.downsample = nn.MaxPool2d(2, 2)

        self.h = 32
        self.w = 32
        self.dim_embd = 256

    def _get_pos_embed(self, H, W):
        return F.interpolate(
            self.pos_embed.reshape(1, self.h, self.w, -1).permute(0, 3, 1, 2),
            size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def generate_weight(self, input):
        feature = self.learnable_degration(input)

        return feature

    def forward(self, x, feature):
        feature = self.generate_weight(x)
        ori = x
        features = []
        enter = self.enter(x)
        en_block1, en_block1_feature11 = self.en_block1(enter, feature=feature[0])
        features.append(en_block1)

        c_128to64 = self.c_128to64(en_block1)
        en_block2, en_block2_feature11 = self.en_block2(c_128to64, feature=feature[1])
        features.append(en_block2)

        c_64to32 = self.c_64to32(en_block2)
        bottom_block3, bottom_block3_feature11 = self.bottom_block3(c_64to32, feature=feature[2])
        features.append(bottom_block3)
        c_32to64 = self.c_32to64(bottom_block3)
        de_block2, de_block2_feature11 = self.de_block2(c_32to64, encoder_feature=en_block2_feature11,
                                                        feature=feature[1])
        features.append(de_block2)
        c_64to128 = self.c_64to128(de_block2)
        de_block3, de_block3_feature11 = self.de_block3(c_64to128, encoder_feature=en_block1_feature11,
                                                        feature=feature[0])
        features.append(de_block3)
        exit = self.exit(de_block3)

        return exit + ori


class Net_wo_perception(nn.Module):
    def __init__(self, channel_query_dict, channel_query_dict_pre,
                 length_list,
                 degregation_propagate=True,
                 local_perception=True,
                 global_perception=True,
                 local_degregation_aware_restore_aware=True,
                 local_degregation_aware=True,
                 local_restore_aware=True,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=True,
                 global_restore_aware=True,
                 perception_SFT=True):
        super().__init__()
        self.enter = nn.Sequential(nn.Conv2d(3, channel_query_dict[192], 3, 1, 1))
        self.en_block1 = Block(channel_query_dict[192], channel_query_dict_pre[192], 2, length_list[0], res=(128, 128),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)

        self.c_128to64 = nn.Sequential(nn.Conv2d(channel_query_dict[192], channel_query_dict[192] // 2, kernel_size=3,
                                                 stride=1, padding=1, bias=True), nn.PixelUnshuffle(2))
        self.en_block2 = Block(channel_query_dict[96], channel_query_dict_pre[96], 4, length_list[1], res=(64, 64),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)
        self.c_64to32 = nn.Sequential(nn.Conv2d(channel_query_dict[96], channel_query_dict[96] // 2, kernel_size=3,
                                                stride=1, padding=1, bias=True), nn.PixelUnshuffle(2))
        self.bottom_block3 = Block(channel_query_dict[48], channel_query_dict_pre[48], 8, length_list[2], res=(16, 16),
                                   degregation_propagate=degregation_propagate,
                                   local_perception=local_perception,
                                   global_perception=global_perception,
                                   local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                                   local_degregation_aware=local_degregation_aware,
                                   local_restore_aware=local_restore_aware,
                                   global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                                   global_degregation_aware=global_degregation_aware,
                                   global_restore_aware=global_restore_aware,
                                   perception_SFT=perception_SFT)

        self.c_32to64 = nn.Sequential(nn.Conv2d(channel_query_dict[48], channel_query_dict[48] * 2, kernel_size=3,
                                                stride=1, padding=1, bias=True), nn.PixelShuffle(2))

        self.de_block2 = Block(channel_query_dict[96], channel_query_dict_pre[96], 4, length_list[1], res=(64, 64),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)

        self.c_64to128 = nn.Sequential(nn.Conv2d(channel_query_dict[96], channel_query_dict[96] * 2, kernel_size=3,
                                                 stride=1, padding=1, bias=True), nn.PixelShuffle(2))
        self.de_block3 = Block(channel_query_dict[192], channel_query_dict_pre[192], 2, length_list[0], res=(128, 128),
                               degregation_propagate=degregation_propagate,
                               local_perception=local_perception,
                               global_perception=global_perception,
                               local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                               local_degregation_aware=local_degregation_aware,
                               local_restore_aware=local_restore_aware,
                               global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                               global_degregation_aware=global_degregation_aware,
                               global_restore_aware=global_restore_aware,
                               perception_SFT=perception_SFT)

        self.exit = nn.Sequential(Refinement_TransformerBlock(channel_query_dict[192]),
                                  nn.Conv2d(channel_query_dict[192], 3, 3, 1, 1))

        self.downsample = nn.MaxPool2d(2, 2)

        self.h = 32
        self.w = 32
        self.dim_embd = 256

    def _get_pos_embed(self, H, W):
        return F.interpolate(
            self.pos_embed.reshape(1, self.h, self.w, -1).permute(0, 3, 1, 2),
            size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward(self, x, feature):
        ori = x
        features = []
        enter = self.enter(x)
        en_block1, en_block1_feature11 = self.en_block1(enter)
        features.append(en_block1)

        c_128to64 = self.c_128to64(en_block1)
        en_block2, en_block2_feature11 = self.en_block2(c_128to64)
        features.append(en_block2)

        c_64to32 = self.c_64to32(en_block2)

        bottom_block3, bottom_block3_feature11 = self.bottom_block3(c_64to32)
        features.append(bottom_block3)

        c_32to64 = self.c_32to64(bottom_block3)
        de_block2, de_block2_feature11 = self.de_block2(c_32to64, encoder_feature=en_block2_feature11)
        features.append(de_block2)
        c_64to128 = self.c_64to128(de_block2)
        de_block3, de_block3_feature11 = self.de_block3(c_64to128, encoder_feature=en_block1_feature11)
        features.append(de_block3)
        exit = self.exit(de_block3)
        return exit + ori


# class Net_encoder(nn.Module):
#     def __init__(self, channel_query_dict, length_list,
#                  degregation_propagate=True,
#                  local_perception=True,
#                  global_perception=True,
#                  local_degregation_aware_restore_aware=True,
#                  local_degregation_aware=True,
#                  local_restore_aware=True,
#                  global_degregation_aware_restore_aware=True,
#                  global_degregation_aware=True,
#                  global_restore_aware=True,
#                  perception_SFT=True):
#         super().__init__()
#         self.enter = nn.Sequential(nn.Conv2d(3, channel_query_dict[128], 3, 1, 1))
#         self.en_block1 = Block(channel_query_dict[128], 8, length_list[0], res=(128, 128),
#                                degregation_propagate=degregation_propagate,
#                                local_perception=local_perception,
#                                global_perception=global_perception,
#                                local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
#                                local_degregation_aware=local_degregation_aware,
#                                local_restore_aware=local_restore_aware,
#                                global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
#                                global_degregation_aware=global_degregation_aware,
#                                global_restore_aware=global_restore_aware,
#                                perception_SFT=perception_SFT)
#         self.downsample1 = nn.Conv2d(channel_query_dict[128], channel_query_dict[128], 3, 2, 1)
#         self.c_128to64 = nn.Conv2d(channel_query_dict[128], channel_query_dict[64], 3, 1, 1)
#         self.en_block2 = Block(channel_query_dict[64], 8, length_list[1], res=(64, 64),
#                                degregation_propagate=degregation_propagate,
#                                local_perception=local_perception,
#                                global_perception=global_perception,
#                                local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
#                                local_degregation_aware=local_degregation_aware,
#                                local_restore_aware=local_restore_aware,
#                                global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
#                                global_degregation_aware=global_degregation_aware,
#                                global_restore_aware=global_restore_aware,
#                                perception_SFT=perception_SFT)
#         self.downsample2 = nn.Conv2d(channel_query_dict[64], channel_query_dict[64], 3, 2, 1)
#         self.c_64to32 = nn.Conv2d(channel_query_dict[64], channel_query_dict[32], 3, 1, 1)
#         # self.en_block3 = Block(channel_query_dict[32], 8, length_list[2], res=(32, 32),
#         #                        degregation_propagate=degregation_propagate,
#         #                        local_perception=local_perception,
#         #                        global_perception=global_perception,
#         #                        local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
#         #                        local_degregation_aware=local_degregation_aware,
#         #                        local_restore_aware=local_restore_aware,
#         #                        global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
#         #                        global_degregation_aware=global_degregation_aware,
#         #                        global_restore_aware=global_restore_aware,
#         #                        perception_SFT=perception_SFT)
#         # self.c_32to16 = nn.Conv2d(channel_query_dict[32], channel_query_dict[16], 3, 1, 1)
#         self.bottom_block3 = Block(channel_query_dict[32], 8, length_list[3], res=(16, 16),
#                                    degregation_propagate=degregation_propagate,
#                                    local_perception=local_perception,
#                                    global_perception=global_perception,
#                                    local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
#                                    local_degregation_aware=local_degregation_aware,
#                                    local_restore_aware=local_restore_aware,
#                                    global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
#                                    global_degregation_aware=global_degregation_aware,
#                                    global_restore_aware=global_restore_aware,
#                                    perception_SFT=perception_SFT)
#         #
#         # self.c_16to32 = nn.Conv2d(channel_query_dict[32], channel_query_dict[64], 3, 1, 1)
#         # self.de_block1 = Block(channel_query_dict[32], 8, length_list[2], res=(32, 32),
#         #                        degregation_propagate=degregation_propagate,
#         #                        local_perception=local_perception,
#         #                        global_perception=global_perception,
#         #                        local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
#         #                        local_degregation_aware=local_degregation_aware,
#         #                        local_restore_aware=local_restore_aware,
#         #                        global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
#         #                        global_degregation_aware=global_degregation_aware,
#         #                        global_restore_aware=global_restore_aware,
#         #                        perception_SFT=perception_SFT)
#         self.c_32to64 = nn.Conv2d(channel_query_dict[32], channel_query_dict[64], 3, 1, 1)
#         self.de_block2 = Block(channel_query_dict[64], 8, length_list[1], res=(64, 64),
#                                degregation_propagate=degregation_propagate,
#                                local_perception=local_perception,
#                                global_perception=global_perception,
#                                local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
#                                local_degregation_aware=local_degregation_aware,
#                                local_restore_aware=local_restore_aware,
#                                global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
#                                global_degregation_aware=global_degregation_aware,
#                                global_restore_aware=global_restore_aware,
#                                perception_SFT=perception_SFT)
#         self.c_64to128 = nn.Conv2d(channel_query_dict[64], channel_query_dict[128], 3, 1, 1)
#         self.de_block3 = Block(channel_query_dict[128], 8, length_list[0], res=(128, 128),
#                                degregation_propagate=degregation_propagate,
#                                local_perception=local_perception,
#                                global_perception=global_perception,
#                                local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
#                                local_degregation_aware=local_degregation_aware,
#                                local_restore_aware=local_restore_aware,
#                                global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
#                                global_degregation_aware=global_degregation_aware,
#                                global_restore_aware=global_restore_aware,
#                                perception_SFT=perception_SFT)
#
#         self.exit = nn.Sequential(Refinement_TransformerBlock(channel_query_dict[128]), nn.Conv2d(channel_query_dict[128], 3, 3, 1, 1))
#
#         self.downsample = nn.MaxPool2d(2, 2)
#
#         self.h = 32
#         self.w = 32
#         self.dim_embd = 256
#
#     def _get_pos_embed(self, H, W):
#         return F.interpolate(
#             self.pos_embed.reshape(1, self.h, self.w, -1).permute(0, 3, 1, 2),
#             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)
#
#     def forward(self, x, feature):
#         ori = x
#         features = []
#         enter = self.enter(x)
#         en_block1, en_block1_feature11 = self.en_block1(enter, feature=feature[0])
#         features.append(en_block1)
#         b1, c1, h1, w1 = en_block1.size()
#         downsample1 = self.downsample1(en_block1)
#
#         c_128to64 = self.c_128to64(downsample1)
#         en_block2, en_block2_feature11 = self.en_block2(c_128to64, feature=feature[1])
#         features.append(en_block2)
#         b2, c2, h2, w2 = en_block2.size()
#         downsample2 = self.downsample2(en_block2)
#
#         c_64to32 = self.c_64to32(downsample2)
#         en_block3, en_block3_feature11 = self.en_block3(c_64to32, feature=feature[2])
#         features.append(en_block3)
#         b3, c3, h3, w3 = en_block3.size()
#         downsample3 = self.downsample(en_block3)
#
#         c_32to16 = self.c_32to16(downsample3)
#         bottom_block3, bottom_block3_feature11 = self.bottom_block3(c_32to16, feature=feature[3])
#         features.append(bottom_block3)
#
#         upsample1 = F.upsample(bottom_block3, (h3, w3))
#         c_16to32 = self.c_16to32(upsample1)
#         de_block1, de_block1_feature11 = self.de_block1(c_16to32, encoder_feature=en_block3_feature11)
#         features.append(de_block1)
#         upsample2 = F.upsample(de_block1, (h2, w2))
#         c_32to64 = self.c_32to64(upsample2)
#         de_block2, de_block2_feature11 = self.de_block2(c_32to64, encoder_feature=en_block2_feature11)
#         features.append(de_block2)
#         upsample3 = F.upsample(de_block2, (h1, w1))
#         c_64to128 = self.c_64to128(upsample3)
#         de_block3, de_block3_feature11 = self.de_block3(c_64to128, encoder_feature=en_block1_feature11)
#         features.append(de_block3)
#         exit = self.exit(de_block3)
#
#         return exit + ori


@ARCH_REGISTRY.register()
class FeMaSRNet(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 scale_factor=1,
                 use_semantic_loss=False,
                 use_residual=True,
                 length_list=[16, 8, 8, 6],
                 Net_type='encoder_decoder',  # encoder_decoder, encoder, wo_perception
                 degregation_propagate=True,
                 local_perception=True,
                 global_perception=True,
                 local_degregation_aware_restore_aware=True,
                 local_degregation_aware=False,
                 local_restore_aware=False,
                 global_degregation_aware_restore_aware=True,
                 global_degregation_aware=False,
                 global_restore_aware=False,
                 perception_SFT=False,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0].astype(int)
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)

        # print('self.codebook_scale', self.codebook_scale)
        #
        # print('codebook_emb_num', codebook_emb_num)
        # print('codebook_emb_dim', codebook_emb_dim)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = scale_factor if LQ_stage else 1
        self.use_residual = use_residual

        # channel_query_dict = {
        #     8: 256,
        #     16: 256,
        #     32: 256,
        #     64: 128,
        #     128: 64,
        #     256: 32,
        #     512: 32,
        # }
        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 128,
            128: 64,
            256: 32,
            512: 32,
        }

        channel_query_dict_lq = {
            8: 256,
            16: 256,
            32: 256,
            64: 192,
            128: 96,
            256: 48,
            512: 32,
        }

        length_list = length_list

        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))  ## 3 ##
        encode_depth = int(np.log2(gt_resolution // self.scale_factor // self.codebook_scale[0]))

        # build derain network Net_type='encoder_decoder', # encoder_decoder, encoder, wo_perception
        if Net_type == 'encoder_decoder':
            self.restoration_network = Net_encoder_decoder(channel_query_dict=channel_query_dict_lq,
                                                           channel_query_dict_pre=channel_query_dict,
                                                           length_list=length_list,
                                                           degregation_propagate=degregation_propagate,
                                                           local_perception=local_perception,
                                                           global_perception=global_perception,
                                                           local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                                                           local_degregation_aware=local_degregation_aware,
                                                           local_restore_aware=local_restore_aware,
                                                           global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                                                           global_degregation_aware=global_degregation_aware,
                                                           global_restore_aware=global_restore_aware,
                                                           perception_SFT=perception_SFT)
        elif Net_type == 'encoder':
            self.restoration_network = Net_encoder(channel_query_dict=channel_query_dict_lq,
                                                   length_list=length_list,
                                                   degregation_propagate=degregation_propagate,
                                                   local_perception=local_perception,
                                                   global_perception=global_perception,
                                                   local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                                                   local_degregation_aware=local_degregation_aware,
                                                   local_restore_aware=local_restore_aware,
                                                   global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                                                   global_degregation_aware=global_degregation_aware,
                                                   global_restore_aware=global_restore_aware,
                                                   perception_SFT=perception_SFT)
        elif Net_type == 'wo_perception':
            self.restoration_network = Net_wo_perception(channel_query_dict=channel_query_dict_lq,
                                                         channel_query_dict_pre=channel_query_dict,
                                                         length_list=length_list,
                                                         degregation_propagate=degregation_propagate,
                                                         local_perception=local_perception,
                                                         global_perception=global_perception,
                                                         local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                                                         local_degregation_aware=local_degregation_aware,
                                                         local_restore_aware=local_restore_aware,
                                                         global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                                                         global_degregation_aware=global_degregation_aware,
                                                         global_restore_aware=global_restore_aware,
                                                         perception_SFT=perception_SFT)
        elif Net_type == 'encoder_decoder_learnable':
            self.restoration_network = Encoder_decoder_learnable(channel_query_dict=channel_query_dict_lq,
                                                                 channel_query_dict_pre=channel_query_dict,
                                                                 length_list=length_list,
                                                                 in_channel=in_channel,
                                                                 encode_depth=encode_depth,
                                                                 gt_res=self.gt_res,
                                                                 norm_type=norm_type,
                                                                 act_type=act_type,
                                                                 degregation_propagate=degregation_propagate,
                                                                 local_perception=local_perception,
                                                                 global_perception=global_perception,
                                                                 local_degregation_aware_restore_aware=local_degregation_aware_restore_aware,
                                                                 local_degregation_aware=local_degregation_aware,
                                                                 local_restore_aware=local_restore_aware,
                                                                 global_degregation_aware_restore_aware=global_degregation_aware_restore_aware,
                                                                 global_degregation_aware=global_degregation_aware,
                                                                 global_restore_aware=global_restore_aware,
                                                                 perception_SFT=perception_SFT)
        print('perception_SFT', perception_SFT)
        # channel_query_dict, length_list, in_channel, encode_depth, gt_res, norm_type, act_type, encoder_decoder_learnable

        self.multiscale_encoder = MultiScaleEncoder(
            in_channel,
            encode_depth,
            self.gt_res // self.scale_factor,
            channel_query_dict,
            norm_type, act_type, LQ_stage
        )

        self.decoder_group = nn.ModuleList()

        for i in range(self.max_depth):
            res = gt_resolution // 2 ** self.max_depth * 2 ** i

            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.decoder_group.append(nn.Sequential(ResBlock(out_ch, out_ch, norm_type, act_type),
                                                ResBlock(out_ch, out_ch, norm_type, act_type)))
        #
        self.out_conv = nn.Sequential(nn.Conv2d(out_ch, 3, 3, 1, 1))

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
                LQ_stage=self.LQ_stage,
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim[scale]
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]

            # print('scale',scale)
            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))

        # semantic loss for HQ pretrain stage
        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss:
            # print('fdsfsdf fsfdsfdsdsfsfds')
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1, 0),
                nn.ReLU(),
            )
            self.vgg_feat_layer = 'relu4_4'
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer])

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def VQGAN(self, input, gt_indices=None):
        out_img, codebook_loss, feature, indices_list = [], [], [], []
        enc_feats = self.multiscale_encoder(input.detach())
        enc_feats = enc_feats[::-1]
        # codebook_loss_list = []
        # indices_list = []
        # quant_idx = 0
        # prev_dec_feat = None
        # prev_quant_feat = None
        # x = enc_feats[0]
        # for i in range(self.max_depth + 1):
        #     cur_res = self.gt_res // 2 ** self.max_depth * 2 ** i
        #     if cur_res in self.codebook_scale:  # needs to perform quantize
        #         if prev_dec_feat is not None:
        #             before_quant_feat = torch.cat((enc_feats[i], prev_dec_feat), dim=1)
        #         else:
        #             before_quant_feat = enc_feats[i]
        #         feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)
        #         feature.append(feat_to_quant)
        #         if gt_indices is not None:
        #             z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant,
        #                                                                              gt_indices[quant_idx])
        #         else:
        #             z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant)
        #
        #         if not self.use_quantize:
        #             z_quant = feat_to_quant
        #         after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)
        #         codebook_loss_list.append(codebook_loss)
        #         indices_list.append(indices)
        #         quant_idx += 1
        #         prev_quant_feat = z_quant
        #         x = after_quant_feat
        #     else:
        #         x = x
        #     x = self.decoder_group[i](x)
        #     prev_dec_feat = x
        # out_img = self.out_conv(x)
        # codebook_loss = sum(codebook_loss_list)
        feature = enc_feats[::-1]
        return input, feature

    def encode_and_decode(self, input, feature=None, current_iter=None):
        out_img, feature_degradation = [], []
        restoration = []
        # if lq_equalize is not None:
        #     ill_radio = input / (lq_equalize + 0.0000001)
        #     structure = self.get_residue_structure_mean(input)
        #     structure = torch.cat([structure, structure, structure], dim=1)
        #     input_equalize = lq_equalize
        # self.print_network(self.restoration_network)
        if self.LQ_stage is False:
            out_img, feature_degradation = self.VQGAN(input)
        else:
            if feature is not None:
                restoration = self.restoration_network(input, feature)
            else:
                restoration = self.restoration_network(input)

        return out_img, feature_degradation, restoration

    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    def check_image_size(self, x, window_size=16):
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % (window_size)) % (
            window_size)
        mod_pad_w = (window_size - w % (window_size)) % (
            window_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
        return x

    @torch.no_grad()
    def test(self, input, feature=None):
        _, _, h_old, w_old = input.shape

        # input = self.check_image_size(input)
        # if lq_equalize is not None:
        #     lq_equalize = self.check_image_size(lq_equalize)

        out_img, feature_degradation, restoration = self.encode_and_decode(input, feature=feature)

        output = restoration
        # output = output[:,:, :h_old, :w_old]

        # self.use_semantic_loss = org_use_semantic_loss
        return output

    def forward(self, input, feature=None):
        # print('**********************************************************')
        # print('input',input.size())
        # print('###########################################################')

        # if gt_indices is not None:
        #     # in LQ training stage, need to pass GT indices for supervise.
        #     out_img_structure, feature_structure, out_img_ill_radio, feature_ill_radio, out_img_input_equalize, feature_input_equalize, enhanced, feature_enhanced = self.encode_and_decode(input, lq_equalize=lq_equalize, deep_feature=deep_feature)
        # else:
        # in HQ stage, or LQ test stage, no GT indices needed.
        out_img, feature_degradation, restoration = self.encode_and_decode(input, feature=feature)

        return out_img, feature_degradation, restoration
