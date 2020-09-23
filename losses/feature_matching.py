import torch
from torch import nn
import torch.nn.functional as F

from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--fem_loss_type', type=str, default='l1', help='l1|mse')
        parser.add('--fem_loss_weight', type=float, default=10.)

    def __init__(self, args):
        super(LossWrapper, self).__init__()
        # Supported loss functions
        losses = {
            'mse': F.mse_loss,
            'l1': F.l1_loss}

        self.loss = losses[args.fem_loss_type]
        self.weight = args.fem_loss_weight

    def forward(self, data_dict, losses_dict):
        real_feats_gen = data_dict['real_feats_gen']
        fake_feats_gen = data_dict['fake_feats_gen']

        # Calculate the loss
        loss = 0
        for real_feats, fake_feats in zip(real_feats_gen, fake_feats_gen):
            loss += self.loss(fake_feats, real_feats.detach())
        loss /= len(real_feats_gen)
        loss *= self.weight

        losses_dict['G_FM'] = loss

        return losses_dict
