import torch
from torch import nn
import torch.nn.functional as F
import math
import functools



############################################################
# PixelUnShuffle layer from https://github.com/cszn/FFDNet #
# Should be removed after it is implemented in PyTorch     #
############################################################

def pixel_unshuffle(inputs, upscale_factor):
    batch_size, channels, in_height, in_width = inputs.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = inputs.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        return pixel_unshuffle(inputs, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

############################################################
#                      Adaptive layers                     #
############################################################

class AdaptiveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(AdaptiveConv2d, self).__init__()
        # Set options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.finetuning = False # set to True by prep_adanorm_for_finetuning method
        
    def forward(self, inputs):
        # Cast parameters into inputs.dtype
        if inputs.type() != self.weight.type():
            self.weight = self.weight.type(inputs.type())
            self.bias = self.bias.type(inputs.type())

        # Reshape parameters into inputs shape
        if self.weight.shape[0] != inputs.shape[0]:
            b = self.weight.shape[0]
            t = inputs.shape[0] // b
            weight = self.weight[:, None].repeat(1, t, 1, 1, 1, 1).view(b*t, *self.weight.shape[1:])
            bias = self.bias[:, None].repeat(1, t, 1).view(b*t, self.bias.shape[1])

        else:
            weight = self.weight
            bias = self.bias

        # Apply convolution
        if self.kernel_size > 1:
            outputs = []
            for i in range(inputs.shape[0]):
                outputs.append(F.conv2d(inputs[i:i+1], weight[i], bias[i], 
                                        self.stride, self.padding, self.dilation, self.groups))
            outputs = torch.cat(outputs, 0)

        else:
            b, c, h, w = inputs.shape
            weight = weight[:, :, :, 0, 0].transpose(1, 2)
            outputs = torch.bmm(inputs.view(b, c, -1).transpose(1, 2), weight).transpose(1, 2).view(b, -1, h, w)
            outputs = outputs + bias[..., None, None]

        return outputs

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        
        if self.padding != 0:
            s += ', padding={padding}'

        if self.dilation != 1:
            s += ', dilation={dilation}'
        
        if self.groups != 1:
            s += ', groups={groups}'
        
        return s.format(**self.__dict__)

class AdaptiveBias(nn.Module):
    def __init__(self, num_features, spatial_size, weight):
        super(AdaptiveBias, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, num_features, spatial_size, spatial_size))

        self.conv = AdaptiveConv2d(num_features, num_features, 1, 1, 0)

        # Init biases
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        b = self.conv.weight.shape[0]
        bias = torch.cat([self.bias]*b)
        bias = self.conv(bias)

        if b != inputs.shape[0]:
            n = inputs.shape[0] // b
            inputs = inputs.view(b, n, *inputs.shape[1:])
            outputs = inputs + self.bias[:, None, :]
            outputs = outputs.view(b*n, *outputs.shape[2:])

        else:
            outputs = inputs + self.bias

        return outputs


class AdaptiveNorm2d(nn.Module):
    def __init__(self, num_features, spatial_size, norm_layer_type, eps=1e-4):
        super(AdaptiveNorm2d, self).__init__()
        # Set options
        self.num_features = num_features
        self.spatial_size = spatial_size
        self.norm_layer_type = norm_layer_type
        self.finetuning = False # set to True by prep_adanorm_for_finetuning method

        if 'spade' in self.norm_layer_type:
            self.pixel_feats = nn.Parameter(torch.empty(1, num_features, spatial_size, spatial_size))
            nn.init.kaiming_uniform_(self.pixel_feats, a=math.sqrt(5))

            self.conv_weight = AdaptiveConv2d(num_features, num_features, 1, 1, 0)
            self.conv_bias = AdaptiveConv2d(num_features, num_features, 1, 1, 0)

        # Supported normalization layers
        norm_layers = {
            'bn': lambda num_features: nn.BatchNorm2d(num_features, eps=eps, affine=False),
            'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=eps, affine=False),
            'none': lambda num_features: nn.Identity()}

        self.norm_layer = norm_layers[self.norm_layer_type.replace('spade_', '')](num_features)
        
    def forward(self, inputs):
        outputs = self.norm_layer(inputs)

        if 'spade' in self.norm_layer_type:
            b = self.conv_weight.weight.shape[0]
            pixel_feats = torch.cat([self.pixel_feats]*b)
            self.weight = self.conv_weight(pixel_feats) + 1.0
            self.bias = self.conv_bias(pixel_feats)

        if len(self.weight.shape) == 2:
            self.weight = self.weight[..., None, None]
            self.bias = self.bias[..., None, None]

        if outputs.type() != self.weight.type():
            # Cast parameters into outputs.dtype
            self.weight = self.weight.type(outputs.type())
            self.bias = self.bias.type(outputs.type())

        if self.weight.shape[0] != outputs.shape[0]:
            b = self.weight.shape[0]
            n = outputs.shape[0] // b
            outputs = outputs.view(b, n, *outputs.shape[1:])
            outputs = outputs * self.weight[:, None] + self.bias[:, None]
            outputs = outputs.view(b*n, *outputs.shape[2:])

        else:
            outputs = outputs * self.weight + self.bias

        return outputs

############################################################
#                      Utility layers                      #
############################################################

class SPADE(nn.Module):
    def __init__(self, num_features, spatial_size, norm_layer_type, eps):
        super(SPADE, self).__init__()
        self.norm_layer_type = norm_layer_type
        self.pixel_feats = nn.Parameter(torch.empty(1, num_features, spatial_size, spatial_size))
        nn.init.kaiming_uniform_(self.pixel_feats, a=math.sqrt(5))

        # Init biases
        self.conv_weight = nn.Conv2d(num_features, num_features, 1, 1, 0)
        self.conv_bias = nn.Conv2d(num_features, num_features, 1, 1, 0)

        # Supported normalization layers
        norm_layers = {
            'bn': lambda num_features: nn.BatchNorm2d(num_features, eps=eps, affine=False),
            'in': lambda num_features: nn.InstanceNorm2d(num_features, eps=eps, affine=False),
            'none': lambda num_features: nn.Identity()}

        self.norm_layer = norm_layers[self.norm_layer_type](num_features)

    def forward(self, inputs):
        outputs = self.norm_layer(inputs)

        weight = self.conv_weight(self.pixel_feats) + 1.0
        bias = self.conv_bias(self.pixel_feats)

        return outputs * weight + bias


class Flatten(nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super(Flatten, self).__init__()
        self.flatten = lambda input: torch.flatten(input, start_dim, end_dim)

    def forward(self, inputs):
        return self.flatten(inputs)


class PixelwiseBias(nn.Module):
    def __init__(self, num_features, spatial_size, weight):
        super(PixelwiseBias, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, num_features, spatial_size, spatial_size))

        # Init biases
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        return inputs + self.bias


class StochasticBias(nn.Module):
    def __init__(self, num_features, spatial_size, weight):
        super(StochasticBias, self).__init__()
        self.spatial_size = spatial_size
        self.scales = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, inputs):
        noise = torch.randn(inputs.shape[0], 1, self.spatial_size, self.spatial_size)

        if noise.type() != inputs.type():
            noise = noise.type(inputs.type())
        
        return inputs + self.scales * noise


def init_weights(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()

############################################################
#                Definitions for the layers                #
############################################################

# Supported activations
activations = {
    'relu': nn.ReLU,
    'leakyrelu': functools.partial(nn.LeakyReLU, negative_slope=0.2)}

# Supported upsampling layers
upsampling_layers = {
    'nearest': lambda stride: nn.Upsample(scale_factor=stride, mode='nearest'),
    'bilinear': lambda stride: nn.Upsample(scale_factor=stride, mode='bilinear'),
    'pixelshuffle': nn.PixelShuffle}

# Supported downsampling layers
downsampling_layers = {
    'avgpool': nn.AvgPool2d,
    'maxpool': nn.MaxPool2d,
    'pixelunshuffle': PixelUnShuffle}

# Supported normalization layers
norm_layers = {
    'none': nn.Identity,
    'bn': lambda num_features, spatial_size, eps: nn.BatchNorm2d(num_features, eps, affine=True),
    'bn_1d': lambda num_features, spatial_size, eps: nn.BatchNorm1d(num_features, eps, affine=True),
    'in': lambda num_features, spatial_size, eps: nn.InstanceNorm2d(num_features, eps, affine=True),
    'ada_bn': functools.partial(AdaptiveNorm2d, norm_layer_type='bn'),
    'ada_in': functools.partial(AdaptiveNorm2d, norm_layer_type='in'),
    'ada_none': functools.partial(AdaptiveNorm2d, norm_layer_type='none'),
    'spade_bn': functools.partial(SPADE, norm_layer_type='bn'),
    'spade_in': functools.partial(SPADE, norm_layer_type='in'),
    'ada_spade_in': functools.partial(AdaptiveNorm2d, norm_layer_type='spade_in'),
    'ada_spade_bn': functools.partial(AdaptiveNorm2d, norm_layer_type='spade_bn')}

# Supported layers for skip connections
skip_layers = {
    'conv': nn.Conv2d,
    'ada_conv': AdaptiveConv2d}

# Supported layers for pixelwise biases
pixelwise_bias_layers = {
    'stochastic': StochasticBias,
    'fixed': PixelwiseBias,
    'adaptive': AdaptiveBias}

############################################################
# Residual block is the base class used for upsampling and #
# downsampling operations inside the networks              #
############################################################

class ResBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, # Parameters for the convolutions
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1,
        dilation: int = 1, 
        groups: int = 1, 
        eps: float = 1e-4, # Used in normalization layers
        spatial_size: int = 1, # Spatial size of the first tensor
        activation_type: str = 'relu',
        norm_layer_type: str = 'none',
        resize_layer_type: str = 'none',
        pixelwise_bias_type: str = 'none', # If not 'none', pixelwise bias is used after each conv
        skip_layer_type: str = 'conv', # Type of convolution in skip connections
        separable_conv: bool = False, # Use separable convolutions
        efficient_upsampling: bool = False, # Place upsampling layer after the first convolution
        first_norm_is_not_adaptive: bool = False, # Force standard normalization in the first norm layer
        return_feats: bool = False, # Output features taken after the first convolution
        return_first_feats: bool = False, # Output additional features taken after the first activation
        few_shot_aggregation: bool = False, # Aggregate few-shot training data in skip connection via mean
        frames_per_person: int = 1, # Number of frames per one person in a batch
        output_aggregated: bool = False, # Output aggregated features
    ) -> nn.Sequential:
        """This is a base module for preactivation residual blocks"""
        super(ResBlock, self).__init__()
        ### Set options for the block ###
        self.return_feats = return_feats
        self.return_first_feats = return_first_feats
        self.few_shot_aggregation = few_shot_aggregation
        self.num_frames = frames_per_person
        self.output_aggregated = output_aggregated

        channel_bias = pixelwise_bias_type == 'none'
        pixelwise_bias = pixelwise_bias_type != 'none'

        normalize = norm_layer_type != 'none'

        upsample = resize_layer_type in upsampling_layers
        downsample = resize_layer_type in downsampling_layers

        ### Set used layers ###
        if pixelwise_bias:
            pixelwise_bias_layer = pixelwise_bias_layers[pixelwise_bias_type]

        activation = activations[activation_type]

        if normalize:
            norm_layer_1 = norm_layers[norm_layer_type if not first_norm_is_not_adaptive else norm_layer_type.replace('ada_', '').replace('spade_', '')]
            norm_layer_2 = norm_layers[norm_layer_type]

        if upsample:
            resize_layer = upsampling_layers[resize_layer_type]

        if downsample:
            resize_layer = downsampling_layers[resize_layer_type]

        skip_layer = skip_layers[skip_layer_type]

        ### Initialize the layers of the first half of the block ###
        layers = []

        if normalize:
            layers += [norm_layer_1(in_channels, spatial_size, eps=eps)]

        layers += [activation(inplace=normalize)] # inplace is set to False if it is the first layer

        if self.return_first_feats:
            self.block_first_feats = nn.Sequential(*layers)

            layers = []

        if upsample and not efficient_upsampling:
            layers += [resize_layer(stride)]

            if spatial_size != 1: spatial_size *= 2

        layers += [nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=channel_bias and not separable_conv)]

        if separable_conv:
            layers += [nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=1,
                bias=channel_bias)]

        if pixelwise_bias:
            layers += [pixelwise_bias_layer(out_channels, spatial_size, layers[-1].weight)]

        if normalize:
            layers += [norm_layer_2(out_channels, spatial_size, eps=eps)]
        
        layers += [activation(inplace=True)]

        self.block_feats = nn.Sequential(*layers)

        ### And initialize the second half ###
        layers = []
        
        if upsample and efficient_upsampling:
            layers += [resize_layer(stride)]

            if spatial_size != 1: spatial_size *= 2

        layers += [nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=channel_bias and not separable_conv)]

        if separable_conv:
            layers += [nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=1,
                bias=channel_bias)]

        if pixelwise_bias:
            layers += [pixelwise_bias_layer(out_channels, spatial_size, layers[-1].weight)]

        if downsample:
            layers += [resize_layer(stride)]

        self.block = nn.Sequential(*layers)

        ### Initialize a skip connection block, if needed ###
        if in_channels != out_channels or upsample or downsample:
            layers = []

            if upsample:
                layers += [resize_layer(stride)]

            layers += [skip_layer(
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=1)]

            if downsample:
                layers += [resize_layer(stride)]
            
            self.skip = nn.Sequential(*layers)

        else:
            self.skip = nn.Identity()

    def forward(self, inputs):
        feats = []

        if hasattr(self, 'block_first_feats'):
            feats += [self.block_first_feats(inputs)]

            outputs = feats[-1]

        else:
            outputs = inputs

        feats += [self.block_feats(outputs)]

        outputs_main = self.block(feats[-1])

        if self.few_shot_aggregation:
            n = self.num_frames
            b = outputs_main.shape[0] // n

            outputs_main = outputs_main.view(b, n, *outputs_main.shape[1:]).mean(dim=1, keepdims=True) # aggregate
            outputs_main = torch.cat([outputs_main]*n, dim=1).view(b*n, *outputs_main.shape[2:]) # repeat

        outputs_skip = self.skip(inputs)

        outputs = outputs_main + outputs_skip

        if self.output_aggregated:
            n = self.num_frames
            b = outputs.shape[0] // n

            outputs = outputs.view(b, n, *outputs.shape[1:]).mean(dim=1) # aggregate

        if self.return_feats: 
            outputs = [outputs, feats]

        return outputs