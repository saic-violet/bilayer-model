import torch
from torch import nn
import torch.nn.functional as F

from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--adv_pred_type', type=str, default='ragan', choices=['gan', 'rgan', 'ragan'])
        parser.add('--adv_loss_weight', type=float, default=0.5)

    def __init__(self, args):
        super(LossWrapper, self).__init__()
        # Supported prediction functions
        get_preds = {
            'gan'  : lambda real_scores, fake_scores: 
                (real_scores, fake_scores),
            'rgan' : lambda real_scores, fake_scores: 
                (real_scores - fake_scores, 
                 fake_scores - real_scores),
            'ragan': lambda real_scores, fake_scores: 
                (real_scores - fake_scores.mean(),
                 fake_scores - real_scores.mean())}

        self.get_preds = get_preds[args.adv_pred_type]

        # The only (currently) supported loss type is hinge loss
        self.loss_dis  = lambda real_preds, fake_preds: torch.relu(1 - real_preds).mean() + torch.relu(1 + fake_preds).mean()
        if 'r' in args.adv_pred_type:
            self.loss_gen = lambda real_preds, fake_preds: torch.relu(1 - fake_preds).mean() + torch.relu(1 + real_preds).mean()
        else:
            self.loss_gen = lambda real_preds, fake_preds: -fake_preds.mean()

        self.weight = args.adv_loss_weight

    def forward(self, data_dict, losses_dict):    
        # Calculate loss for dis
        real_scores = data_dict['real_scores']
        fake_scores = data_dict['fake_scores_dis']
        real_preds, fake_preds = self.get_preds(real_scores, fake_scores)
        losses_dict['D_ADV'] = self.loss_dis(real_preds, fake_preds) * self.weight
            
        # Calculate loss for gen
        real_scores = real_scores.detach()
        fake_scores = data_dict['fake_scores_gen']
        real_preds, fake_preds = self.get_preds(real_scores, fake_scores)
        losses_dict['G_ADV'] = self.loss_gen(real_preds, fake_preds) * self.weight

        return losses_dict