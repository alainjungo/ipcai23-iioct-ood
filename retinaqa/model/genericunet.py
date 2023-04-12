from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import helpers as helper


def get_norm_builder(norm: str, dims: int, **norm_args):
    if norm == 'bn':
        if dims == 1:
            return DefaultClassBuilder(nn.BatchNorm1d, **norm_args)
        elif dims == 2:
            return DefaultClassBuilder(nn.BatchNorm2d, **norm_args)
        elif dims == 3:
            return DefaultClassBuilder(nn.BatchNorm3d, **norm_args)
    if norm == 'inst':
        if dims == 1:
            return DefaultClassBuilder(nn.InstanceNorm1d, **norm_args)
        elif dims == 2:
            return DefaultClassBuilder(nn.InstanceNorm2d, **norm_args)
        elif dims == 3:
            return DefaultClassBuilder(nn.InstanceNorm3d, **norm_args)
    raise NotImplementedError(f'unsupported norm="{norm}" with dims={dims}')


def get_activation_builder(act: str, **act_args):
    if act == 'relu':
        return DefaultClassBuilder(nn.ReLU, **act_args)
    elif act == 'lrelu':
        return DefaultClassBuilder(nn.LeakyReLU, **act_args)
    raise NotImplementedError(f'unsupported activation="{act}"')


def get_pool_builder(pool: str, dims: int, **pool_args):
    if pool == 'max':
        if dims == 1:
            return DefaultClassBuilder(nn.MaxPool1d, **pool_args)
        elif dims == 2:
            return DefaultClassBuilder(nn.MaxPool2d, **pool_args)
        elif dims == 3:
            return DefaultClassBuilder(nn.MaxPool3d, **pool_args)
    elif pool == 'avg':
        if dims == 1:
            return DefaultClassBuilder(nn.AvgPool1d, **pool_args)
        elif dims == 2:
            return DefaultClassBuilder(nn.AvgPool2d, **pool_args)
        elif dims == 3:
            return DefaultClassBuilder(nn.AvgPool3d, **pool_args)
    raise NotImplementedError(f'unsupported pooling="{pool}" with dims={dims}')


def get_conv_builder(dims: int, **conv_args):
    if dims == 1:
        return DefaultClassBuilder(nn.Conv1d, **conv_args)
    elif dims == 2:
        return DefaultClassBuilder(nn.Conv2d, **conv_args)
    elif dims == 3:
        return DefaultClassBuilder(nn.Conv3d, **conv_args)
    raise NotImplementedError(f'unsupported conv with dims={dims}')


def get_do_builder(dims: int, **do_args):
    if dims == 1:
        return DefaultClassBuilder(nn.Dropout, **do_args)
    elif dims == 2:
        return DefaultClassBuilder(nn.Dropout2d, **do_args)
    elif dims == 3:
        return DefaultClassBuilder(nn.Dropout3d, **do_args)
    raise NotImplementedError(f'unsupported dropout with dims={dims}')


def get_interpolation_builder(dims: int, **intpl_args):
    if dims == 1:
        return DefaultClassBuilder(helper.InterpolateWrapper, mode='linear', **intpl_args)
    elif dims == 2:
        return DefaultClassBuilder(helper.InterpolateWrapper, mode='bilinear', **intpl_args)
    elif dims == 3:
        DefaultClassBuilder(helper.InterpolateWrapper, mode='trilinear', **intpl_args)
    raise NotImplementedError(f'unsupported interpolation mode with dims={dims}')


def build_conv_norm_act(in_ch, out_ch, build_conv, build_norm, build_activation, build_dropout, info: dict):
    seq = nn.Sequential()
    seq.add_module('conv', build_conv(in_ch, out_ch, info=info))
    do = build_dropout(info=info)
    if do is not None:
        seq.add_module('dropout', do)
    seq.add_module('norm', build_norm(out_ch, info=info))
    seq.add_module('activation', build_activation(info=info))
    return seq


def get_block(depth, channels: list, build_conv, build_norm, build_activation, build_dropout):
    block = nn.Sequential()
    for i in range(len(channels) - 1):
        info = {'depth': depth, 'rep_idx': i, 'repetitions': len(channels) - 1}
        part = build_conv_norm_act(channels[i], channels[i + 1], build_conv, build_norm, build_activation, build_dropout, info)
        block.add_module(f'conv_norm_act_{i}', part)
    return block


class DownBlock(nn.Module):

    def __init__(self, block, pool):
        super().__init__()
        self.block = block
        self.pool = pool

    def forward(self, x):
        skip_x = self.block(x)
        down_x = self.pool(skip_x)
        return down_x, skip_x


class UpBlock(nn.Module):

    def __init__(self, block, upsample, concat):
        super().__init__()
        self.upsample = upsample
        self.concat = concat
        self.block = block

    def forward(self, up_x, skip_x):
        up_x = self.upsample(up_x)
        x = self.concat(up_x, skip_x)
        x = self.block(x)
        return x


class Condition:

    def __call__(self, info: dict):
        raise NotImplementedError()


class PositionalCondition(Condition):

    def __init__(self, mode='all') -> None:
        super().__init__()
        assert mode in ('all', 'none', 'first', 'last')
        self.mode = mode

    def __call__(self, info: dict):
        if self.mode == 'all':
            return True
        if self.mode == 'first' and info['rep_idx'] == 0:
            return True
        if self.mode == 'last' and info['rep_idx'] == info['repetitions'] - 1:
            return True
        else:
            return False


class DefaultClassBuilder:

    def __init__(self, cls, condition=None, **cls_args) -> None:
        super().__init__()
        self.cls = cls
        self.cls_args = cls_args
        self.condition = condition

    def __call__(self, *args, info: dict = None, **kwargs):
        # additional_params also enables overwriting exising params
        params = {**self.cls_args}
        for k, v in kwargs.items():
            params[k] = v
        if self.condition is None or self.condition(info):
            return self.cls(*args, **params)
        else:
            return None


class GenericUnet(nn.Module):

    def __init__(self, out_ch, in_ch, channels=(16, 32, 64, 128, 256), repetitions=2, dropout=0.1, dims=1,
                 kernel_size: Union[int, tuple] = 3, reduction: Union[int, tuple] = 2):
        super().__init__()
        self.dims = dims

        down_channels = [in_ch] + list(channels)[:-1]
        up_channels = [channels[0]] + list(channels)[:-1]
        bottom_channels = channels[-1]

        pad = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
        build_conv = get_conv_builder(dims, padding=pad, kernel_size=kernel_size)
        build_norm = get_norm_builder('bn', dims)
        build_activation = get_activation_builder('relu', inplace=True)

        # build_dropout = get_do_builder(dims, p=dropout) if dropout is not None else lambda: None
        build_dropout = get_do_builder(dims, p=dropout)
        build_pool = get_pool_builder('max', dims, kernel_size=reduction)
        build_upsample = get_interpolation_builder(dims, scale_factor=reduction)
        build_concat = DefaultClassBuilder(PaddedConcat, dims=dims)

        self.down_blocks = nn.ModuleList()
        depth = 0

        build_dropout.condition = PositionalCondition('last' if dropout is not None else 'none')
        while depth < len(channels) - 1:
            block_channels = [down_channels[depth]] + repetitions*[down_channels[depth + 1]]
            block = get_block(depth, block_channels, build_conv, build_norm, build_activation, build_dropout)
            down_block = DownBlock(block, build_pool())
            self.down_blocks.append(down_block)
            depth += 1

        build_dropout.condition = PositionalCondition('none')
        block_channels = [down_channels[depth]] + (repetitions - 1)*[bottom_channels] + [up_channels[depth]]
        self.bottom_block = get_block(depth, block_channels, build_conv, build_norm, build_activation, build_dropout)

        build_dropout.condition = PositionalCondition('first' if dropout is not None else 'none')
        self.up_blocks = nn.ModuleList()
        while depth > 0:
            depth -= 1
            block_channels = [up_channels[depth+1] + down_channels[depth+1]] + (repetitions - 1) * [up_channels[depth + 1]] + [up_channels[depth]]
            block = get_block(depth, block_channels, build_conv, build_norm, build_activation, build_dropout)
            up_block = UpBlock(block, build_upsample(), build_concat())
            self.up_blocks.append(up_block)

        self.conv_cls = build_conv(in_channels=up_channels[depth], out_channels=out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        skip_xs = []
        for down_block in self.down_blocks:
            x, skip_x = down_block(x)
            skip_xs.append(skip_x)

        x = self.bottom_block(x)

        for inv_depth, up_block in enumerate(self.up_blocks, 1):
            skip_x = skip_xs[-inv_depth]
            x = up_block(x, skip_x)

        out_logits = self.conv_cls(x)
        return out_logits


class PaddedConcat(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, up_x, skip_x):
        up_x = pad_if_required(up_x, skip_x, self.dims)
        x = torch.cat((up_x, skip_x), 1)
        return x


def pad_if_required(up_x, skip_x, dims):
    up_shape, skip_shape = up_x.size()[-dims:], skip_x.size()[-dims:]
    if up_shape < skip_shape:
        padding = []
        for dim in range(dims):
            diff = skip_shape[-dim] - up_shape[-dim]
            padding = [diff // 2, diff // 2 + (diff % 2)] + padding
        up_x = F.pad(up_x, padding)
    return up_x


def get_features(unet: GenericUnet, x):
    mean_dims = tuple(range(-unet.dims, 0))

    features = []
    skip_xs = []
    for down_block in unet.down_blocks:
        x, skip_x = down_block(x)
        skip_xs.append(skip_x)

        feature_mean = skip_x.mean(dim=mean_dims).detach()
        features.append(feature_mean)

    x = unet.bottom_block(x)
    feature_mean = x.mean(dim=mean_dims).detach()
    features.append(feature_mean)

    for inv_depth, up_block in enumerate(unet.up_blocks, 1):
        skip_x = skip_xs[-inv_depth]
        x = up_block(x, skip_x)

        feature_mean = x.mean(dim=mean_dims).detach()
        features.append(feature_mean)

    out = unet.conv_cls(x)
    feature_mean = out.mean(dim=mean_dims).detach()
    features.append(feature_mean)

    return features, out

