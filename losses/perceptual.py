import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import pathlib

# This project
from runners import utils as rn_utils
from networks import utils as nt_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        # Extractor parameters
        parser.add('--per_full_net_names', type=str, default='vgg19_imagenet_pytorch, vgg16_face_caffe')
        parser.add('--per_net_layers', type=str, default='1,6,11,20,29; 1,6,11,18,25', help='a list of layers indices')
        parser.add('--per_pooling', type=str, default='avgpool', choices=['maxpool', 'avgpool'])
        parser.add('--per_loss_apply_to', type=str, default='pred_target_imgs_lf_detached, target_imgs')

        # Loss parameters
        parser.add('--per_loss_type', type=str, default='l1')
        parser.add('--per_loss_weights', type=str, default='10.0, 0.01')
        parser.add('--per_layer_weights', type=str, default='0.03125, 0.0625, 0.125, 0.25, 1.0')
        parser.add('--per_loss_names', type=str, default='VGG19, VGGFace')

    def __init__(self, args):
        super(LossWrapper, self).__init__()
        ### Define losses ###
        losses = {
            'mse': F.mse_loss,
            'l1': F.l1_loss}

        self.loss = losses[args.per_loss_type]

        # Weights for each feature extractor
        self.weights = rn_utils.parse_str_to_list(args.per_loss_weights, value_type=float, sep=',')
        self.layer_weights = rn_utils.parse_str_to_list(args.per_layer_weights, value_type=float, sep=',')
        self.names = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.per_loss_names, sep=';')]

        ### Define extractors ###
        self.apply_to = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.per_loss_apply_to, sep=';')]
        weights_dir = pathlib.Path(args.project_dir) / 'pretrained_weights' / 'perceptual'

        # Architectures for the supported networks 
        networks = {
            'vgg16': models.vgg16,
            'vgg19': models.vgg19}

        # Build a list of used networks
        self.nets = nn.ModuleList()
        self.full_net_names = rn_utils.parse_str_to_list(args.per_full_net_names, sep=',')

        for full_net_name in self.full_net_names:
            net_name, dataset_name, framework_name = full_net_name.split('_')

            if dataset_name == 'imagenet' and framework_name == 'pytorch':
                self.nets.append(networks[net_name](pretrained=True))
                mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None] * 2 - 1
                std  = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None] * 2
            
            elif framework_name == 'caffe':
                self.nets.append(networks[net_name]())
                self.nets[-1].load_state_dict(torch.load(weights_dir / f'{full_net_name}.pth'))
                self.nets[-1] = self.nets[-1]
                mean = torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 127.5 - 1
                std  = torch.FloatTensor([     1.,      1.,      1.])[None, :, None, None] / 127.5
            
            # Register means and stds as buffers
            self.register_buffer(f'{full_net_name}_mean', mean)
            self.register_buffer(f'{full_net_name}_std', std)

        # Perform the slicing according to the required layers
        for n, (net, block_idx) in enumerate(zip(self.nets, rn_utils.parse_str_to_list(args.per_net_layers, sep=';'))):
            net_blocks = nn.ModuleList()

            # Parse indices of slices
            block_idx = rn_utils.parse_str_to_list(block_idx, value_type=int, sep=',')
            for i, idx in enumerate(block_idx):
                block_idx[i] = idx

            # Slice conv blocks
            layers = []
            for i, layer in enumerate(net.features):
                if layer.__class__.__name__ == 'MaxPool2d' and args.per_pooling == 'avgpool':
                    layer = nn.AvgPool2d(2)
                layers.append(layer)
                if i in block_idx:
                    net_blocks.append(nn.Sequential(*layers))
                    layers = []

            # Add layers for prediction of the scores (if needed)
            if block_idx[-1] == 'fc':
                layers.extend([
                    nn.AdaptiveAvgPool2d(7),
                    utils.Flatten(1)])
                for layer in net.classifier:
                    layers.append(layer)
                net_blocks.append(nn.Sequential(*layers))

            # Store sliced net
            self.nets[n] = net_blocks

    def forward(self, data_dict, losses_dict):
        for i, (tensor_name, target_tensor_name) in enumerate(self.apply_to):
            # Extract inputs
            real_imgs = data_dict[target_tensor_name]
            fake_imgs = data_dict[tensor_name]

            # Prepare inputs
            b, t, c, h, w = real_imgs.shape
            real_imgs = real_imgs.view(-1, c, h, w)
            fake_imgs = fake_imgs.view(-1, c, h, w)

            with torch.no_grad():
                real_feats_ext = self.forward_extractor(real_imgs)

            fake_feats_ext = self.forward_extractor(fake_imgs)

            # Calculate the loss
            for n in range(len(self.names[i])):
                loss = 0
                for real_feats, fake_feats, weight in zip(real_feats_ext[n], fake_feats_ext[n], self.layer_weights):
                    loss += self.loss(fake_feats, real_feats.detach()) * weight
                loss *= self.weights[n]

                losses_dict[f'G_{self.names[i][n]}'] = loss

        return losses_dict

    def forward_extractor(self, imgs):
        # Calculate features
        feats = []
        for net, full_net_name in zip(self.nets, self.full_net_names):
            # Preprocess input image
            mean = getattr(self, f'{full_net_name}_mean')
            std = getattr(self, f'{full_net_name}_std')
            feats.append([(imgs - mean) / std])

            # Forward pass through blocks
            for block in net:
                feats[-1].append(block(feats[-1][-1]))

            # Remove input image
            feats[-1].pop(0)

        return feats
