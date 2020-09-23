# Third party
import torch
from torch import nn
import torch.nn.functional as F

# This project
from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--seg_loss_type', type=str, default='bce')
        parser.add('--seg_loss_weights', type=float, default=10.)
        parser.add('--seg_loss_apply_to', type=str, default='pred_target_inf_segs_logits, target_segs', help='can specify multiple tensor names from data_dict')
        parser.add('--seg_loss_names', type=str, default='BCE', help='name for each loss')

    def __init__(self, args):
        super(LossWrapper, self).__init__()   
        self.apply_to = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.seg_loss_apply_to, sep=';')]
        self.names = rn_utils.parse_str_to_list(args.seg_loss_names, sep=',')

        # Supported loss functions
        losses = {
            'bce': F.binary_cross_entropy_with_logits,
            'dice': lambda fake_seg, real_seg: torch.log((fake_seg**2).sum() + (real_seg**2).sum()) - torch.log((2 * fake_seg * real_seg).sum())}

        self.loss = losses[args.seg_loss_type]

        self.weights = args.seg_loss_weights

        self.eps = args.eps

    def forward(self, data_dict, losses_dict):
        for i, (tensor_name, target_tensor_name) in enumerate(self.apply_to):
            real_segs = data_dict[target_tensor_name]
            fake_segs = data_dict[tensor_name]

            b, t = fake_segs.shape[:2]
            fake_segs = fake_segs.view(b*t, *fake_segs.shape[2:])

            if 'HalfTensor' in fake_segs.type():  
                real_segs = real_segs.type(fake_segs.type())

            real_segs = real_segs.view(b*t, *real_segs.shape[2:])

            losses_dict['G_' + self.names[i]] = self.loss(fake_segs, real_segs) * self.weights

        return losses_dict