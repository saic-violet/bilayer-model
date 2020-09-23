# Third party
import torch
from torch import nn
import torch.nn.functional as F

# This project
from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--wpr_loss_type', type=str, default='l1')
        parser.add('--wpr_loss_weight', type=float, default=10.0)
        parser.add('--wpr_loss_weight_decay', type=float, default=0.9, help='multiplicative decay of loss weight')
        parser.add('--wpr_loss_decay_schedule', type=int, default=50, help='num iters after which decay happends')
        parser.add('--wpr_loss_apply_to', type=str, default='pred_target_delta_uvs', help='tensors this loss is applied to')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.apply_to = rn_utils.parse_str_to_list(args.wpr_loss_apply_to)
        self.eps = args.eps

        self.reg_type = args.wpr_loss_type
        self.weight = args.wpr_loss_weight

        self.weight_decay = args.wpr_loss_weight_decay
        self.decay_schedule = args.wpr_loss_decay_schedule
        self.num_iters = 0

    def forward(self, data_dict, losses_dict):
        if self.num_iters == self.decay_schedule:
            self.weight = max(self.weight * self.weight_decay, self.eps)
            self.num_iters = 1

        if self.weight == self.eps:
            return losses_dict

        loss = 0

        for tensor_name in self.apply_to:
            if self.reg_type == 'l1':
                loss += data_dict[tensor_name].abs().mean()
            else:
                raise # Unknown reg_type

        loss /= len(self.apply_to)
 
        losses_dict['G_WPR'] = loss * self.weight

        if self.weight_decay != 1.0:
            self.num_iters += 1

        return losses_dict