"""
Discriminator and Generator implementation from Compositional GAN paper
"""

import torch
import torch.nn as nn
from torch import index_select
import torch.nn.functional as F
import functools
import Config


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (4, 4), stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Spatial Transformer Network Discriminator
class Discriminator_STN(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=(4, 4), stride=(1, 1), padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return nn.Sigmoid()(x)


class Discriminator_Decomp(nn.Module):
    def __init__(self, channels_img):
        super(Discriminator_Decomp, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 128 x 128
            nn.Conv2d(
                channels_img, 64, kernel_size=4, stride=2, padding=1
            ),  # 64 x 64
            nn.LeakyReLU(0.2),
            self._block(64, 64 * 2, 4, 2, 1),  # 32 x 32
            self._block(64 * 2, 64 * 4, 4, 2, 1),  # 16 x 16
            self._block(64 * 4, 64 * 8, 4, 2, 1),  # 8 x 8
            self._block(64 * 8, 64 * 16, 4, 2, 1),  # 4 x 4
            nn.Conv2d(64 * 16, 1, kernel_size=4, stride=2, padding=0),  # 1 x 1
        )

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),  # LayerNorm <-> InstanceNorm
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        xs = self.disc(x)
        return nn.Sigmoid()(xs)


class Discriminator_Comp(nn.Module):
    def __init__(self, channels):
        super(Discriminator_Comp, self).__init__()
        norm_layer = get_norm_layer()
        self.disc = NLayerDiscriminator(channels, n_layers=3, norm_layer=norm_layer)

    def forward(self, x):
        xs = self.disc(x)
        return nn.Sigmoid()(xs)


class CompositionGenerator(nn.Module):
    def __init__(self, channels):
        super(CompositionGenerator, self).__init__()
        norm_layer = get_norm_layer()
        self.comp_gen = ResnetGeneratorconv(3*channels, channels, norm_layer=norm_layer, use_dropout=False, n_blocks=9)

    def forward(self, x):
        return self.comp_gen(x)


class HierarchyGenerator(nn.Module):
    def __init__(self):
        super(HierarchyGenerator, self).__init__()
        self.densenet121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
        self.hierarchy = nn.Sequential(
            self.densenet121,
            nn.Linear(1000, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.hierarchy(x)


class LogoStyleGenerator(nn.Module):
    def __init__(self):
        super(LogoStyleGenerator, self).__init__()
        self.densenet161 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=False)
        self.logo_style = nn.Sequential(
            self.densenet161,
            nn.Linear(1000, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.logo_style(x)


class DecompositionGenerator(nn.Module):
    def __init__(self, channels):
        super(DecompositionGenerator, self).__init__()
        norm_layer = get_norm_layer()
        self.decomp_gen = ResnetGeneratorconv(channels, 3*channels, norm_layer=norm_layer, use_dropout=False, n_blocks=9)

    def forward(self, x):
        return self.decomp_gen(x)


class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        self.stn = SpatialTransformer()

    def forward(self, x):
        return self.stn(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

class ResnetGeneratorconv(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[0], padding_type='reflect', noise=False, y_x=1):
        assert (n_blocks >= 0)
        super(ResnetGeneratorconv, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.noise = noise
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 128 #256
        model1 = [nn.ReflectionPad2d(3),  # 134 #262
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),  # 128 #256
                  norm_layer(ngf),
                  nn.ReLU(True)]

        mult = 1
        if y_x == 2:
            model1 += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=(1, 3),
                                 stride=(1, 2), padding=(0, 1), bias=use_bias),  # 128 #128
                       norm_layer(ngf * mult),
                       nn.ReLU(True)]

        model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                             stride=2, padding=1, bias=use_bias),  # 64 #64
                   norm_layer(ngf * mult * 2),
                   nn.ReLU(True)]

        mult = 2
        if noise:
            nc_in = ngf * mult + int(ngf * mult / 4.0)
        else:
            nc_in = ngf * mult
        model2 = [nn.Conv2d(nc_in, ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=use_bias),  # 32 #64
                  norm_layer(ngf * mult * 2),
                  nn.ReLU(True)]

        n_downsampling = 2
        # mode: nearest, bilinear, bicubic
        mode = 'bilinear'
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model2 += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model2 += [nn.Upsample(scale_factor=2, mode=mode, align_corners=True),
                       nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]
        if y_x == 2:
            model2 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 2),
                                          kernel_size=(1, 3), stride=(1, 2),
                                          padding=(0, 1), output_padding=(0, 1),
                                          bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]

        model2 += [nn.ReflectionPad2d(3)]
        model2 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model2 += [nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def forward(self, input):
        x1 = nn.parallel.data_parallel(self.model1, input, self.gpu_ids)
        return nn.parallel.data_parallel(self.model2, x1, self.gpu_ids)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[0]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        # 128 256
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),  # 63 128
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Relative Spatial Transformer Network
class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(7, 7)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(24, 30, kernel_size=(5, 5)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(30, 90, kernel_size=(5, 5)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(90, 90, kernel_size=(5, 5)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 4 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(90 * 4 * 4, 96),
            nn.ReLU(True),
            nn.Linear(96, 3 * 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 90 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 6, 3)

        theta_1 = index_select(theta, 1, Config.inp1)
        theta_2 = index_select(theta, 1, Config.inp2)
        theta_3 = index_select(theta, 1, Config.inp3)

        input_1 = index_select(x, 1, Config.ind1)
        input_2 = index_select(x, 1, Config.ind2)
        input_3 = index_select(x, 1, Config.ind3)

        grid_1 = F.affine_grid(theta_1, input_1.size(), align_corners=True)
        grid_2 = F.affine_grid(theta_2, input_2.size(), align_corners=True)
        grid_3 = F.affine_grid(theta_3, input_3.size(), align_corners=True)

        x1 = F.grid_sample(input_1, grid_1, padding_mode="border", align_corners=True)
        x2 = F.grid_sample(input_2, grid_2, padding_mode="border", align_corners=True)
        x3 = F.grid_sample(input_3, grid_3, padding_mode="border", align_corners=True)

        return x1, x2, x3
