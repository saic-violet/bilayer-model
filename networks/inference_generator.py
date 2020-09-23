# Third party
import torch
from torch import nn
import torch.nn.functional as F
import math

# This project
from runners import utils as rn_utils
from networks import utils as nt_utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--inf_num_channels',         default=32, type=int, 
                                                 help='minimum number of channels')

        parser.add('--inf_max_channels',         default=256, type=int, 
                                                 help='maximum number of channels')

        parser.add('--inf_pred_segmentation',    default='True', type=rn_utils.str2bool, choices=[True, False],
                                                 help='set inference generator to output a segmentation mask')

        parser.add('--inf_norm_layer_type',      default='ada_bn', type=str,
                                                 help='norm layer inside the inference generator')

        parser.add('--inf_input_tensor_size',    default=4, type=int, 
                                                 help='input spatial size of the convolutional part')

        parser.add('--inf_activation_type',      default='leakyrelu', type=str,
                                                 help='activation layer inside the generators')

        parser.add('--inf_upsampling_type',      default='nearest', type=str,
                                                 help='upsampling layer inside the generator')

        parser.add('--inf_skip_layer_type',      default='ada_conv', type=str,
                                                 help='skip connection layer type')

        parser.add('--inf_pred_source_data',     default='False', type=rn_utils.str2bool, choices=[True, False], 
                                                 help='predict inference generator outputs for the source data')

        parser.add('--inf_calc_grad',            default='False', type=rn_utils.str2bool, choices=[True, False], 
                                                 help='force gradients calculation in the generator')

        parser.add('--inf_apply_masks',          default='True', type=rn_utils.str2bool, choices=[True, False], 
                                                 help='apply segmentation masks to predicted and ground-truth images')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        # Initialize options
        self.args = args

        # Generator
        self.gen_inf = Generator(args)

        # Projector (prediction of adaptive parameters)
        self.prj_inf = Projector(args)

        # Greate a meshgrid, which is used for UVs calculation from deltas
        grid = torch.linspace(-1, 1, args.image_size + 1)
        grid = (grid[1:] + grid[:-1]) / 2
        v, u = torch.meshgrid(grid, grid)
        identity_grid = torch.stack([u, v], 2)[None] # 1 x h x w x 2
        self.register_buffer('identity_grid', identity_grid)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:

        # Do not store activations if this network is not being trained
        if 'inference_generator' not in networks_to_train and not self.args.inf_calc_grad:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        ### Prepare inputs ###
        idt_embeds = data_dict['source_idt_embeds']
        target_pose_embeds = data_dict['target_pose_embeds']

        if self.args.inf_apply_masks and self.args.inf_pred_segmentation:
            # Predicted segmentation masks are applied to target images
            target_imgs = data_dict['target_imgs']

        pred_tex_hf_rgbs = data_dict['pred_tex_hf_rgbs'][:, 0]

        b, t = target_pose_embeds.shape[:2]
        target_pose_embeds = target_pose_embeds.view(b*t, *target_pose_embeds.shape[2:])

        if self.args.inf_apply_masks and self.args.inf_pred_segmentation:
            target_imgs = target_imgs.view(b*t, *target_imgs.shape[2:])

        if self.args.inf_pred_source_data:
            source_pose_embeds = data_dict['source_pose_embeds']

            n = source_pose_embeds.shape[1]
            source_pose_embeds = source_pose_embeds.view(b*n, *source_pose_embeds.shape[2:])

        ### Forward through the projectors ###
        inf_weights, inf_biases = self.prj_inf(idt_embeds)
        self.assign_adaptive_params(self.gen_inf, inf_weights, inf_biases)

        ### Forward target poses through the inference generator ###
        outputs = self.gen_inf(target_pose_embeds)

        # Parse the outputs
        pred_target_delta_uvs = outputs[0]
        pred_target_uvs = self.identity_grid + pred_target_delta_uvs.permute(0, 2, 3, 1)

        pred_target_delta_lf_rgbs = outputs[1]

        if self.args.inf_pred_segmentation:
            pred_target_segs_logits = outputs[2]
            pred_target_segs = torch.sigmoid(pred_target_segs_logits)

        ### Forward source poses through the inference generator (if needed) ###
        if self.args.inf_pred_source_data:
            outputs = self.gen_inf(source_pose_embeds)

            # Parse the outputs
            source_delta_uvs = outputs[0]
            pred_source_uvs = self.identity_grid + source_delta_uvs.permute(0, 2, 3, 1)

            pred_source_delta_lf_rgbs = outputs[1]

            if self.args.inf_pred_segmentation:
                pred_source_segs_logits = outputs[2]
                pred_source_segs = torch.sigmoid(pred_source_segs_logits)

        ### Combine components into an output target image
        pred_tex_hf_rgbs_repeated = torch.cat([pred_tex_hf_rgbs[:, None]]*t, dim=1)
        pred_tex_hf_rgbs_repeated = pred_tex_hf_rgbs_repeated.view(b*t, *pred_tex_hf_rgbs.shape[1:])

        pred_target_delta_hf_rgbs = F.grid_sample(pred_tex_hf_rgbs_repeated, pred_target_uvs)

        # Final image
        pred_target_imgs = pred_target_delta_lf_rgbs + pred_target_delta_hf_rgbs

        if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
            # Get an image with a low-frequency component detached
            pred_target_imgs_lf_detached = pred_target_delta_lf_rgbs.detach() + pred_target_delta_hf_rgbs

        # Mask output images (if needed)
        if self.args.inf_apply_masks and self.args.inf_pred_segmentation:
            pred_target_masks = pred_target_segs.detach()

            target_imgs = target_imgs * pred_target_masks + (-1) * (1 - pred_target_masks)

            pred_target_imgs = pred_target_imgs * pred_target_masks + (-1) * (1 - pred_target_masks)

            pred_target_delta_lf_rgbs = pred_target_delta_lf_rgbs * pred_target_masks + (-1) * (1 - pred_target_masks)

            if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
                pred_target_imgs_lf_detached = pred_target_imgs_lf_detached * pred_target_masks + (-1) * (1 - pred_target_masks)

        if 'inference_generator' not in networks_to_train and not self.args.inf_calc_grad:
            torch.set_grad_enabled(prev)

        ### Store outputs ###
        reshape_target_data = lambda data: data.view(b, t, *data.shape[1:])
        reshape_source_data = lambda data: data.view(b, n, *data.shape[1:])

        data_dict['pred_target_imgs'] = reshape_target_data(pred_target_imgs)
        if self.args.inf_pred_segmentation:
            data_dict['pred_target_segs'] = reshape_target_data(pred_target_segs)

        # Output debugging results
        data_dict['pred_target_uvs'] = reshape_target_data(pred_target_uvs)
        data_dict['pred_target_delta_lf_rgbs'] = reshape_target_data(pred_target_delta_lf_rgbs)
        data_dict['pred_target_delta_hf_rgbs'] = reshape_target_data(pred_target_delta_hf_rgbs)

        # Output results needed for training
        if 'inference_generator' in networks_to_train or self.args.inf_calc_grad:
            data_dict['pred_target_delta_uvs'] = reshape_target_data(pred_target_delta_uvs)
            data_dict['pred_target_imgs_lf_detached'] = reshape_target_data(pred_target_imgs_lf_detached)

            if self.args.inf_pred_segmentation:
                data_dict['pred_target_segs_logits'] = reshape_target_data(pred_target_segs_logits)

        if self.args.inf_apply_masks and self.args.inf_pred_segmentation:
            data_dict['target_imgs'] = reshape_target_data(target_imgs)

        # Output results related to source imgs (if needed)
        if self.args.inf_pred_source_data:
            data_dict['pred_source_uvs'] = reshape_source_data(pred_source_uvs)
            data_dict['pred_source_delta_lf_rgbs'] = reshape_source_data(pred_source_delta_lf_rgbs)

            if self.args.inf_pred_segmentation:
                data_dict['pred_source_segs'] = reshape_source_data(pred_source_segs)

        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        visuals = []

        if self.args.inf_pred_source_data:
            # Predicted source LF rgbs
            visuals += [data_dict['pred_source_delta_lf_rgbs']]

            # Predicted source HF rgbs
            if 'pred_source_delta_hf_rgbs' in data_dict.keys():
                visuals += [data_dict['pred_source_delta_hf_rgbs']]

            # Predicted source UVs
            pred_source_uvs = data_dict['pred_source_uvs'].permute(0, 3, 1, 2)

            b, _, h, w = pred_source_uvs.shape
            pred_source_uvs = torch.cat([
                    pred_source_uvs, 
                    torch.empty(b, 1, h, w, dtype=pred_source_uvs.dtype, device=pred_source_uvs.device).fill_(-1)
                ], 
                dim=1)

            visuals += [torch.cat([pred_source_uvs])]

            # Predicted source segs
            if self.args.inf_pred_segmentation:
                pred_source_segs = data_dict['pred_source_segs']

                visuals += [torch.cat([(pred_source_segs - 0.5) * 2] * 3, 1)]

        # Predicted textures
        visuals += [data_dict['pred_tex_hf_rgbs']]

        if 'pred_enh_tex_hf_rgbs' in data_dict.keys():
            # Predicted enhated textures
            visuals += [data_dict['pred_enh_tex_hf_rgbs']]

        # Target images
        visuals += [data_dict['target_imgs']]

        # Predicted images
        visuals += [data_dict['pred_target_imgs']]

        # Predicted enhated images
        if 'pred_enh_target_imgs' in data_dict.keys():
            visuals += [data_dict['pred_enh_target_imgs']]

        # Predicted target LF rgbs
        visuals += [data_dict['pred_target_delta_lf_rgbs']]

        # Predicted target HF rgbs
        visuals += [data_dict['pred_target_delta_hf_rgbs']]

        if 'pred_enh_target_delta_hf_rgbs' in data_dict.keys():
            # Predicted enhated target HF rgbs
            visuals += [data_dict['pred_enh_target_delta_hf_rgbs']]

        # Predicted target UVs
        pred_target_uvs = data_dict['pred_target_uvs'].permute(0, 3, 1, 2)

        b, _, h, w = pred_target_uvs.shape
        pred_target_uvs = torch.cat([
                pred_target_uvs, 
                torch.empty(b, 1, h, w, dtype=pred_target_uvs.dtype, device=pred_target_uvs.device).fill_(-1)
            ], 
            dim=1)

        visuals += [torch.cat([pred_target_uvs])]

        if self.args.inf_pred_segmentation:
            # Target segmentation
            target_segs = data_dict['target_segs']
            visuals += [torch.cat([(target_segs - 0.5) * 2] * 3, 1)]

            # Predicted target segmentation
            pred_target_segs = data_dict['pred_target_segs']
            visuals += [torch.cat([(pred_target_segs - 0.5) * 2] * 3, 1)]

        return visuals

    @staticmethod
    def assign_adaptive_params(net, weights, biases):
        i = 0
        for m in net.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                m.weight = weights[i] + 1.0
                m.bias = biases[i]
                i += 1

            elif m.__class__.__name__ == 'AdaptiveConv2d':
                m.weight = weights[i]
                m.bias = biases[i]
                i += 1

    @staticmethod
    def adaptive_params_mixing(net, indices):
        for m in net.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                m.weight = m.weight[indices]
                m.bias = m.bias[indices]

            elif m.__class__.__name__ == 'AdaptiveConv2d':
                m.weight = m.weight[indices]
                m.bias = m.bias[indices]

    def __repr__(self):
        output = ''

        num_params = 0
        for p in self.prj_inf.parameters():
            num_params += p.numel()
        output += self.prj_inf.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params
        output += '\n'

        num_params = 0
        for p in self.gen_inf.parameters():
            num_params += p.numel()
        output += self.gen_inf.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        # Set options for the blocks
        num_blocks = int(math.log(args.image_size // args.inf_input_tensor_size, 2))
        out_channels = min(int(args.inf_num_channels * 2**num_blocks), args.inf_max_channels)

        # Construct the upsampling blocks
        layers = []

        for i in range(num_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(args.inf_num_channels * 2**i), args.inf_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.inf_activation_type, 
                norm_layer_type=args.inf_norm_layer_type,
                resize_layer_type=args.inf_upsampling_type,
                skip_layer_type=args.inf_skip_layer_type,
                efficient_upsampling=True,
                first_norm_is_not_adaptive=i == num_blocks - 1)]

        norm_layer = nt_utils.norm_layers[args.inf_norm_layer_type]
        activation = nt_utils.activations[args.inf_activation_type]

        layers += [
            norm_layer(out_channels, spatial_size=1, eps=args.eps),
            activation(inplace=True)]

        self.blocks = nn.Sequential(*layers)

        # Get the list of required heads
        heads = [(2, nn.Tanh), (3, nn.Tanh)]

        if args.inf_pred_segmentation:
            heads += [(1, None)]

        # Initialize the heads
        self.heads = nn.ModuleList()

        for num_outputs, final_activation in heads:
            layers = [nn.Conv2d(
                in_channels=out_channels, 
                out_channels=num_outputs, 
                kernel_size=3, 
                stride=1, 
                padding=1)]

            if final_activation is not None:
                layers += [final_activation()]

            self.heads += [nn.Sequential(*layers)]

    def forward(self, inputs):
        outputs = self.blocks(inputs).contiguous()

        results = []

        for head in self.heads:
            results += [head(outputs)]

        return results


class Projector(nn.Module):
    def __init__(self, args, bottleneck_size=1024):
        super(Projector, self).__init__()
        # Calculate parameters of the blocks
        num_blocks = int(math.log(args.image_size // args.inf_input_tensor_size, 2))
        
        # FC channels perform a lowrank matrix decomposition
        self.channel_mults = []
        self.avgpools = nn.ModuleList()
        self.fc_blocks = nn.ModuleList()

        for i in range(num_blocks, 0, -1):
            in_channels = min(int(args.emb_num_channels * 2**i), args.emb_max_channels)

            out_in_channels = min(int(args.inf_num_channels * 2**i), args.inf_max_channels)
            out_channels = min(int(args.inf_num_channels * 2**(i-1)), args.inf_max_channels)

            channel_mult = out_in_channels / float(in_channels)
            self.channel_mults += [channel_mult]

            # Average pooling is applied to embeddings before FC
            s = int(bottleneck_size**0.5 * channel_mult)
            self.avgpools += [nn.AdaptiveAvgPool2d((s, s))]

            # Define decompositions for the i-th block
            self.fc_blocks += [nn.ModuleList()]

            # First AdaNorm weights and biases
            self.fc_blocks[-1] += [
                nn.Sequential(
                    nn.Linear(int(s**2 / channel_mult), in_channels),
                    nn.Linear(in_channels, in_channels),
                    nn.Linear(in_channels, 2))]

            # Skip conv weights and biases
            if args.inf_skip_layer_type == 'ada_conv':
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_in_channels + 1))]

            # Second AdaNorm weights and biases
            self.fc_blocks[-1] += [nn.Sequential(
                nn.Linear(int(s**2 / channel_mult), in_channels),
                nn.Linear(in_channels, in_channels),
                nn.Linear(in_channels, 2))]

    def forward(self, embeds):
        weights = []
        biases = []

        for embed, fc_block, channel_mult, avgpool in zip(embeds, self.fc_blocks, self.channel_mults, self.avgpools):
            b, c, h, w = embed.shape
            embed = avgpool(embed)

            c_out = int(c * channel_mult)
            embed = embed.view(b * c_out, -1)

            for fc in fc_block:
                params = fc(embed)

                params = params.view(b, c_out, -1)
                
                weight = params[:, :, :-1].squeeze()

                if weight.shape[0] != b and len(weight.shape) > 1 or len(weight.shape) > 2:
                    weight = weight[..., None, None] # 1x1 conv weight

                bias = params[:, :, -1].squeeze()

                if b == 1:
                    weight = weight[None]
                    bias = bias[None]
                    
                weights += [weight]
                biases += [bias]

        return weights, biases