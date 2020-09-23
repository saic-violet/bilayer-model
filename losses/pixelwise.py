# Third party
import torch
from torch import nn
import torch.nn.functional as F

# This project
from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--pix_loss_type', type=str, default='l1')
        parser.add('--pix_loss_weights', type=str, default='10.0', help='comma separated floats')
        parser.add('--pix_loss_apply_to', type=str, default='pred_target_delta_lf_rgbs, target_imgs', help='can specify multiple tensor names from data_dict')
        parser.add('--pix_loss_names', type=str, default='L1', help='name for each loss')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.apply_to = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.pix_loss_apply_to, sep=';')]
        
        # Supported loss functions
        losses = {
            'mse': F.mse_loss,
            'l1': F.l1_loss,
            'ce': F.cross_entropy}

        self.loss = losses[args.pix_loss_type]

        # Weights for each feature extractor
        self.weights = rn_utils.parse_str_to_list(args.pix_loss_weights, value_type=float)
        self.names = rn_utils.parse_str_to_list(args.pix_loss_names)

    def forward(self, data_dict, losses_dict):
        for i, (tensor_name, target_tensor_name) in enumerate(self.apply_to):
            real_imgs = data_dict[target_tensor_name]
            fake_imgs = data_dict[tensor_name]

            b, t = fake_imgs.shape[:2]
            fake_imgs = fake_imgs.view(b*t, *fake_imgs.shape[2:])

            if 'HalfTensor' in fake_imgs.type():  
                real_imgs = real_imgs.type(fake_imgs.type())

            real_imgs = real_imgs.view(b*t, *real_imgs.shape[2:])

            loss = self.loss(fake_imgs, real_imgs.detach())

            losses_dict['G_' + self.names[i]] = loss * self.weights[i]

        return losses_dict