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
        parser.add('--emb_num_channels',          default=64, type=int, 
                                                  help='minimum number of channels')

        parser.add('--emb_max_channels',          default=512, type=int, 
                                                  help='maximum number of channels')

        parser.add('--emb_no_stickman',           action='store_true', 
                                                  help='do not input stickman into the embedder')

        parser.add('--emb_output_tensor_size',    default=8, type=int,
                                                  help='spatial size of the last tensor')

        parser.add('--emb_norm_layer_type',       default='none', type=str,
                                                  help='norm layer inside the embedder')

        parser.add('--emb_activation_type',       default='leakyrelu', type=str,
                                                  help='activation layer inside the embedder')

        parser.add('--emb_downsampling_type',     default='avgpool', type=str,
                                                  help='downsampling layer inside the embedder')

        parser.add('--emb_apply_masks',           default='True', type=rn_utils.str2bool, choices=[True, False],
                                                  help='apply segmentation masks to source ground-truth images')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.args = args
        
        self.net = Embedder(args)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        """The function modifies the input data_dict to contain the embeddings for the source images"""

        # Do not store activations if this network is not being trained
        if 'identity_embedder' not in networks_to_train:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        ### Prepare inputs ###
        inputs = data_dict['source_imgs']
        b, n = inputs.shape[:2]

        if self.args.emb_apply_masks:
            inputs = inputs * data_dict['source_segs'] + (-1) * (1 - data_dict['source_segs'])

        if not self.args.emb_no_stickman:
            inputs = torch.cat([inputs, data_dict['source_stickmen']], 2)

        ### Main forward pass ###
        source_embeds = self.net(inputs)

        if 'identity_embedder' not in networks_to_train:
            torch.set_grad_enabled(prev)

        ### Store outputs ###
        data_dict['source_idt_embeds'] = source_embeds

        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        visuals = [data_dict['source_imgs'].detach()]

        if 'source_stickmen' in data_dict.keys():
            visuals += [data_dict['source_stickmen']]
        
        return visuals

    def __repr__(self):
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()
        output = self.net.__repr__()

        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()
        # Number of encoding blocks
        num_enc_blocks = int(math.log(args.image_size // args.emb_output_tensor_size, 2))

        # Number of decoding blocks (which is equal to the number of blocks in the generator)
        num_dec_blocks = int(math.log(args.image_size // args.tex_input_tensor_size, 2))

        ### Source images embedding ###
        out_channels = args.emb_num_channels

        # Construct the encoding blocks
        layers = [
            nn.Conv2d(
                in_channels=3 + 3 * (not args.emb_no_stickman), 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1)]

        for i in range(1, num_enc_blocks + 1):
            in_channels = out_channels
            out_channels = min(int(args.emb_num_channels * 2**i), args.emb_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.emb_activation_type, 
                norm_layer_type=args.emb_norm_layer_type,
                resize_layer_type=args.emb_downsampling_type,
                frames_per_person=args.num_source_frames,
                output_aggregated=i == num_enc_blocks)]

        self.enc = nn.Sequential(*layers)

        # And the decoding blocks
        layers = []

        for i in range(num_dec_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(args.tex_num_channels * 2**i), args.tex_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels,
                eps=args.eps,
                activation_type=args.emb_activation_type, 
                norm_layer_type=args.emb_norm_layer_type,
                resize_layer_type='none',
                return_feats=True)]

        self.dec_blocks = nn.ModuleList(layers)

    def forward(self, inputs):
        b, n, c, h, w = inputs.shape
        outputs = self.enc(inputs.view(-1, c, h, w))
        
        # Obtain embeddings at the final resolution
        embeds = []

        # Produce a stack of embeddings with different channels num
        for block in self.dec_blocks:
            outputs, embeds_block = block(outputs)
            embeds += embeds_block

        # Average over all source images (if needed)
        if embeds[0].shape[0] == b*n:
            embeds = [embeds_block.view(b, n, *embeds_block.shape[1:]).mean(dim=1) for embeds_block in embeds]

        return embeds