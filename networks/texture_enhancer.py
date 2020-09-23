# Third party
import torch
from torch import nn
import torch.nn.functional as F
import math
import time

from runners import utils as rn_utils
from networks import utils as nt_utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--enh_num_channels',            default=64, type=int, 
                                                    help='minimum number of channels')

        parser.add('--enh_max_channels',            default=128, type=int, 
                                                    help='maximum number of channels')

        parser.add('--enh_bottleneck_tensor_size',  default=128, type=int, 
                                                    help='spatial size of the tensor in the bottleneck')

        parser.add('--enh_num_blocks',              default=8, type=int, 
                                                    help='number of convolutional blocks at the bottleneck resolution')

        parser.add('--enh_unrolling_depth',         default=4, type=int, 
                                                    help='number of consequtive unrolling iterations')

        parser.add('--enh_guiding_rgb_loss_type',   default='sse', type=str, choices=['sse', 'l1'],
                                                    help='lightweight loss that guides the enhates of the rgb texture')

        parser.add('--enh_detach_inputs',           default='True', type=rn_utils.str2bool, choices=[True, False],
                                                    help='detach input tensors (for efficient training)')

        parser.add('--enh_norm_layer_type',         default='none', type=str,
                                                    help='norm layer inside the enhancer')

        parser.add('--enh_activation_type',         default='leakyrelu', type=str,
                                                    help='activation layer inside the enhancer')

        parser.add('--enh_downsampling_type',       default='avgpool', type=str,
                                                    help='downsampling layer inside the enhancer')

        parser.add('--enh_upsampling_type',         default='nearest', type=str,
                                                    help='upsampling layer inside the enhancer')

        parser.add('--enh_apply_masks',             default='True', type=rn_utils.str2bool, choices=[True, False],
                                                    help='apply segmentation masks to predicted and ground-truth images')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.args = args

        self.net = Generator(args)

        rgb_losses = {
            'sse': lambda fake, real: ((real - fake)**2).sum() / 2,
            'l1': lambda fake, real: (real - fake).abs().sum()}

        self.rgb_loss = rgb_losses[args.enh_guiding_rgb_loss_type]

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:

        # Do not store activations if this network is not being trained
        if 'texture_enhancer' not in networks_to_train:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        ### Prepare inputs ###
        source_uvs = data_dict['pred_source_uvs']
        source_delta_lf_rgbs = data_dict['pred_source_delta_lf_rgbs']
        source_imgs = data_dict['source_imgs']

        enh_tex_hf_rgbs = data_dict['pred_tex_hf_rgbs'].clone()

        target_uvs = data_dict['pred_target_uvs']
        pred_target_delta_lf_rgbs = data_dict['pred_target_delta_lf_rgbs']
        
        if self.args.enh_apply_masks and self.args.inf_pred_segmentation:
            target_imgs = data_dict['target_imgs']
            pred_target_imgs = data_dict['pred_target_imgs']

        if self.args.inf_pred_segmentation:
            pred_source_segs = data_dict['pred_source_segs']
            pred_target_segs = data_dict['pred_target_segs']

        # Reshape inputs
        b, t, c, h, w = pred_target_delta_lf_rgbs.shape
        n = source_uvs.shape[1]

        source_uvs = source_uvs.view(b*n, h, w, 2)
        source_delta_lf_rgbs = source_delta_lf_rgbs.view(b*n, c, h, w)
        source_imgs = source_imgs.view(b*n, c, h, w)

        enh_tex_hf_rgbs = enh_tex_hf_rgbs[:, 0]

        if self.args.enh_detach_inputs:
            source_uvs = source_uvs.detach()
            source_delta_lf_rgbs = source_delta_lf_rgbs.detach()
            enh_tex_hf_rgbs = enh_tex_hf_rgbs.detach()

        if self.args.enh_apply_masks and self.args.inf_pred_segmentation:
            target_imgs = target_imgs.view(b*t, c, h, w)
            pred_target_imgs = pred_target_imgs.view(b*t, c, h, w)
        
        target_uvs = target_uvs.view(b*t, h, w, 2)
        pred_target_delta_lf_rgbs = pred_target_delta_lf_rgbs.view(b*t, c, h, w)

        if self.args.enh_detach_inputs:
            target_uvs = target_uvs.detach()
            pred_target_delta_lf_rgbs = pred_target_delta_lf_rgbs.detach()

        if self.args.inf_pred_segmentation:
            pred_source_segs = pred_source_segs.view(b*n, 1, h, w)
            pred_target_segs = pred_target_segs.view(b*t, 1, h, w)

            source_imgs = source_imgs * pred_source_segs + (-1) * (1 - pred_source_segs)

            if self.args.enh_detach_inputs:
                pred_source_segs = pred_source_segs.detach()
                pred_target_segs = pred_target_segs.detach()

        for i in range(self.args.enh_unrolling_depth):
            # Calculation of gradients is required for enhancer losses
            prev_enh = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

            # Repeat the texture n times for n train frames
            enh_tex_hf_rgbs_i = torch.cat([enh_tex_hf_rgbs[:, None]]*n, dim=1).view(b*n, *enh_tex_hf_rgbs.shape[1:])
            enh_tex_hf_rgbs_i_grad = enh_tex_hf_rgbs_i.clone().detach()
            enh_tex_hf_rgbs_i_grad.requires_grad = True

            # Current approximation of the source image
            pred_source_imgs = source_delta_lf_rgbs.detach() + F.grid_sample(enh_tex_hf_rgbs_i_grad, source_uvs.detach())

            if self.args.inf_pred_segmentation:
                pred_source_imgs = pred_source_imgs * pred_source_segs + (-1) * (1 - pred_source_segs)

            # Calculate the gradients with respect to the enhancer losses
            loss_enh = self.rgb_loss(pred_source_imgs, source_imgs)

            loss_enh.backward()

            # Forward pass through the enhancer network
            inputs = torch.cat([
                enh_tex_hf_rgbs_i,
                enh_tex_hf_rgbs_i_grad.grad.detach()],
                dim=1)

            torch.set_grad_enabled(prev_enh)

            outputs = self.net(inputs)

            # Update the texture
            delta_enh_tex_hf_rgbs_i = torch.tanh(outputs[:, :3])

            # Aggregate data (if needed)
            if delta_enh_tex_hf_rgbs_i.shape[0] == b*n:
                delta_enh_tex_hf_rgbs_i = delta_enh_tex_hf_rgbs_i.view(b, n, *delta_enh_tex_hf_rgbs_i.shape[1:]).mean(dim=1)

            enh_tex_hf_rgbs = enh_tex_hf_rgbs + delta_enh_tex_hf_rgbs_i

        # Evaluate on real frames
        enh_tex_hf_rgbs = torch.cat([enh_tex_hf_rgbs[:, None]]*t, 1)
        enh_tex_hf_rgbs = enh_tex_hf_rgbs.view(b*t, *enh_tex_hf_rgbs.shape[2:])

        pred_enh_target_delta_hf_rgbs = F.grid_sample(enh_tex_hf_rgbs, target_uvs) # high-freq. component
        pred_enh_target_imgs = pred_target_delta_lf_rgbs + pred_enh_target_delta_hf_rgbs # final image

        if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
            # Get an image with a low-frequency component detached
            pred_enh_target_imgs_lf_detached = pred_target_delta_lf_rgbs.detach() + pred_enh_target_delta_hf_rgbs

        if self.args.inf_pred_segmentation and self.args.enh_apply_masks:
            pred_target_masks = pred_target_segs.detach()

            # Apply segmentation predicted by the main model
            pred_target_imgs = pred_target_imgs * pred_target_segs + (-1) * (1 - pred_target_segs)

            # Apply possbily enhanced segmentation
            target_imgs = target_imgs * pred_target_masks + (-1) * (1- pred_target_masks)
            pred_enh_target_imgs = pred_enh_target_imgs * pred_target_masks + (-1) * (1 - pred_target_masks)

            if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
                pred_enh_target_imgs_lf_detached = pred_enh_target_imgs_lf_detached * pred_target_masks + (-1) * (1 - pred_target_masks)

        if 'texture_enhancer' not in networks_to_train:
            torch.set_grad_enabled(prev)

        ### Store outputs ###
        reshape_target_data = lambda data: data.view(b, t, *data.shape[1:])

        data_dict['pred_enh_tex_hf_rgbs'] = reshape_target_data(enh_tex_hf_rgbs)

        data_dict['pred_enh_target_imgs'] = reshape_target_data(pred_enh_target_imgs)

        # Output debugging results
        data_dict['pred_enh_target_delta_hf_rgbs'] = reshape_target_data(pred_enh_target_delta_hf_rgbs)

        if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
            data_dict['pred_enh_target_imgs_lf_detached'] = reshape_target_data(pred_enh_target_imgs_lf_detached)

        if self.args.enh_apply_masks and self.args.inf_pred_segmentation:
            data_dict['target_imgs'] = reshape_target_data(target_imgs)
            data_dict['pred_target_imgs'] = reshape_target_data(pred_target_imgs)

        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        visuals = []

        return visuals

    def __repr__(self):
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()
        output = self.net.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        num_down_blocks = int(math.log(args.image_size // args.enh_bottleneck_tensor_size, 2))

        # Initialize the residual blocks
        layers = []

        in_channels = 6
        out_channels = args.enh_num_channels

        layers = [nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1)]

        for i in range(1, num_down_blocks + 1):
            in_channels = out_channels
            out_channels = min(int(args.enh_num_channels * 2**i), args.enh_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.enh_activation_type, 
                norm_layer_type=args.enh_norm_layer_type,
                resize_layer_type=args.enh_downsampling_type)]

        for i in range(args.enh_num_blocks):
            layers += [nt_utils.ResBlock(
                in_channels=out_channels, 
                out_channels=out_channels,
                eps=args.eps,
                activation_type=args.enh_activation_type, 
                norm_layer_type=args.enh_norm_layer_type,
                resize_layer_type='none',
                frames_per_person=args.num_source_frames,
                output_aggregated=i == args.enh_num_blocks - 1)]

        for i in range(num_down_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(args.enh_num_channels * 2**i), args.enh_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.enh_activation_type, 
                norm_layer_type=args.enh_norm_layer_type,
                resize_layer_type=args.enh_upsampling_type)]

        in_channels = out_channels
        out_channels = 3

        norm_layer = nt_utils.norm_layers[args.enh_norm_layer_type]
        activation = nt_utils.activations[args.enh_activation_type]

        layers += [
            norm_layer(out_channels),
            activation(inplace=True),
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1)]

        self.blocks = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.blocks(inputs)

        return outputs