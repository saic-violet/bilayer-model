# Third party
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# This project
from networks import utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--pse_num_channels',      default=256, type=int, 
                                              help='number of intermediate channels')

        parser.add('--pse_num_blocks',        default=4, type=int, 
                                              help='number of encoding blocks')

        parser.add('--pse_in_channels',       default=394, type=int,
                                              help='number of channels in either latent pose (if present) of keypoints')

        parser.add('--pse_emb_source_pose',   action='store_true', 
                                              help='predict embeddings for the source pose')

        parser.add('--pse_norm_layer_type',   default='none', type=str,
                                              help='norm layer inside the pose embedder')

        parser.add('--pse_activation_type',   default='leakyrelu', type=str,
                                              help='activation layer inside the pose embedder')

        parser.add('--pse_use_harmonic_enc',  action='store_true', 
                                              help='encode keypoints with harmonics')

        parser.add('--pse_num_harmonics',     default=4, type=int, 
                                              help='number of frequencies used')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.args = args

        self.net = PoseEmbedder(args)

        if self.args.pse_use_harmonic_enc:
            frequencies = torch.ones(args.pse_num_harmonics) * np.pi * 2**torch.arange(args.pse_num_harmonics)
            frequencies = frequencies[None, None]

            self.register_buffer('frequencies', frequencies)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        """The function modifies the input data_dict to contain the pose 
        embeddings for the target and (optinally) source images"""

        # Do not store activations if this network is not being trained
        if 'keypoints_embedder' not in networks_to_train:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        ### Prepare inputs ###
        target_poses = data_dict['target_poses']

        b, t = target_poses.shape[:2]
        target_poses = target_poses.view(b*t, -1)

        # Encode with harmonics (if needed)
        if self.args.pse_use_harmonic_enc:
            target_poses = (target_poses[..., None] * self.frequencies).view(b*t, -1)
            target_poses = torch.cat([torch.sin(target_poses), torch.cos(target_poses)], dim=1)

        if self.args.pse_emb_source_pose:
            source_poses = data_dict['source_poses']

            n = source_poses.shape[1]
            source_poses = source_poses.view(b*n, -1)

            # Encode with harmonics (if needed)
            if self.args.pse_use_harmonic_enc:
                source_poses = (source_poses[..., None] * self.frequencies).view(b*t, -1)
                source_poses = torch.cat([torch.sin(source_poses), torch.cos(source_poses)], dim=1)

        ### Main forward pass ###
        target_embeds = self.net(target_poses)

        if self.args.pse_emb_source_pose:
            source_embeds = self.net(source_poses)

        if 'keypoints_embedder' not in networks_to_train:
            torch.set_grad_enabled(prev)

        ### Store outputs ###
        data_dict['target_pose_embeds'] = target_embeds.view(b, t, *target_embeds.shape[1:])

        if self.args.pse_emb_source_pose:
            data_dict['source_pose_embeds'] = source_embeds.view(b, n, *source_embeds.shape[1:])

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


class PoseEmbedder(nn.Module):
    def __init__(self, args):
        super(PoseEmbedder, self).__init__()
        # Calculate output size of the embedding
        self.num_channels = args.inf_max_channels
        self.spatial_size = args.inf_input_tensor_size

        # Initialize keypoints-encoding MLP
        norm_layer = utils.norm_layers[args.pse_norm_layer_type]
        activation = utils.activations[args.pse_activation_type]

        # Set input number of channels
        if args.pse_use_harmonic_enc:
            in_channels = args.pse_in_channels * args.pse_num_harmonics * 2
        else:
            in_channels = args.pse_in_channels

        # Set latent number of channels
        if args.pse_num_blocks == 1:
            num_channels = self.num_channels * self.spatial_size**2
        else:
            num_channels = args.pse_num_channels

        # Define encoding blocks
        layers = [nn.Linear(in_channels, num_channels)]
        
        for i in range(1, args.pse_num_blocks - 1):
            if args.pse_norm_layer_type != 'none':
                layers += [norm_layer(num_channels, None, eps=args.eps)]

            layers += [
                activation(inplace=True),
                nn.Linear(num_channels, num_channels)]

        if args.pse_num_blocks != 1:
            if args.pse_norm_layer_type != 'none':
                layers += [norm_layer(num_channels, None, eps=args.eps)]

            layers += [
                activation(inplace=True),
                nn.Linear(num_channels, self.num_channels * self.spatial_size**2)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        pose_embeds = self.mlp(inputs)

        pose_embeds = pose_embeds.view(-1, self.num_channels, self.spatial_size, self.spatial_size)

        return pose_embeds
