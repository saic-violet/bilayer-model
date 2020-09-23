# Third party
import torch
from torch import nn
import math

# This project
from networks import utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--dis_num_channels',        default=64, type=int, 
                                                help='minimum number of channels')

        parser.add('--dis_max_channels',        default=512, type=int, 
                                                help='maximum number of channels')

        parser.add('--dis_no_stickman',         action='store_true', 
                                                help='do not input stickman into the discriminator')

        parser.add('--dis_num_blocks',          default=6, type=int, 
                                                help='number of convolutional blocks')

        parser.add('--dis_output_tensor_size',  default=8, type=int, 
                                                help='spatial size of the last tensor')

        parser.add('--dis_norm_layer_type',     default='bn', type=str,
                                                help='norm layer inside the discriminator')

        parser.add('--dis_activation_type',     default='leakyrelu', type=str,
                                                help='activation layer inside the discriminator')

        parser.add('--dis_downsampling_type',   default='avgpool', type=str,
                                                help='downsampling layer inside the discriminator')

        parser.add('--dis_fake_imgs_name',      default='pred_target_imgs', type=str,
                                                help='name of the tensor with fake images')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.args = args
        
        self.net = Discriminator(args)

    def forward(
            self, 
            data_dict: dict,
            net_names_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        
        # Extract inputs
        real_inputs = data_dict['target_imgs']
        fake_inputs = data_dict[self.args.dis_fake_imgs_name]

        # Input stickman (if needed)
        if not self.args.dis_no_stickman:
            real_inputs = torch.cat([real_inputs, data_dict['target_stickmen']], 2)
            fake_inputs = torch.cat([fake_inputs, data_dict['target_stickmen']], 2)

        # Reshape inputs
        b, t, c, h, w = real_inputs.shape
        real_inputs = real_inputs.view(-1, c, h, w)
        fake_inputs = fake_inputs.view(-1, c, h, w)

        ### Perform a dis forward pass ###
        for p in self.parameters():
            p.requires_grad = True

        # Concatenate batches
        inputs = torch.cat([real_inputs, fake_inputs.detach()])

        # Calculate outputs
        scores_dis, _ = self.net(inputs)

        # Split outputs into real and fake
        real_scores, fake_scores_dis = scores_dis.split(b)

        ### Store outputs ###
        data_dict['real_scores'] = real_scores
        data_dict['fake_scores_dis'] = fake_scores_dis

        ### Perform a gen forward pass ###
        for p in self.parameters():
            p.requires_grad = False

        # Concatenate batches
        inputs = torch.cat([real_inputs, fake_inputs])

        # Calculate outputs
        scores_gen, feats_gen  = self.net(inputs)

        # Split outputs into real and fake
        _, fake_scores_gen = scores_gen.split(b)

        feats = [feats_block.split(b) for feats_block in feats_gen]
        real_feats_gen, fake_feats_gen = map(list, zip(*feats))

        ### Store outputs ###
        data_dict['fake_scores_gen'] = fake_scores_gen

        data_dict['real_feats_gen'] = real_feats_gen
        data_dict['fake_feats_gen'] = fake_feats_gen
            
        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        visuals = []
        
        if 'target_stickmen' in data_dict.keys():
            visuals += [data_dict['target_stickmen']]
        
        return visuals

    def __repr__(self):
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()
        output = self.net.__repr__()

        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        # Set options for the blocks
        num_down_blocks = int(math.log(args.image_size // args.dis_output_tensor_size, 2))

        ### Construct the encoding blocks ###
        out_channels = args.dis_num_channels

        # The first block
        self.first_conv = nn.Conv2d(
            in_channels=3 + 3 * (not args.dis_no_stickman), 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1)

        # Downsampling blocks
        self.blocks = nn.ModuleList()

        for i in range(1, num_down_blocks + 1):
            in_channels = out_channels
            out_channels = min(int(args.dis_num_channels * 2**i), args.dis_max_channels)

            self.blocks += [utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                activation_type=args.dis_activation_type, 
                norm_layer_type=args.dis_norm_layer_type,
                resize_layer_type=args.dis_downsampling_type,
                return_feats=True)]

        # And the blocks at the same resolution
        for i in range(num_down_blocks + 1, args.dis_num_blocks + 1):
            self.blocks += [utils.ResBlock(
                in_channels=out_channels, 
                out_channels=out_channels,
                eps=args.eps,
                activation_type=args.dis_activation_type, 
                norm_layer_type=args.dis_norm_layer_type,
                resize_layer_type='none',
                return_feats=True)]

        # Final convolutional block
        norm_layer = utils.norm_layers[args.dis_norm_layer_type]
        activation = utils.activations[args.dis_activation_type]

        self.final_block = nn.Sequential(
            norm_layer(out_channels, None, eps=args.eps),
            activation(inplace=True))

        ### Realism score prediction ###
        self.linear = nn.Conv2d(out_channels, 1, 1)

    def forward(self, inputs):
        # Convolutional part
        conv_outputs = self.first_conv(inputs)

        feats = []
        for block in self.blocks:
            conv_outputs, block_feats = block(conv_outputs)
            feats += block_feats

        conv_outputs = self.final_block(conv_outputs)

        # Linear head
        scores = self.linear(conv_outputs)

        return scores, feats