import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--ssm_use_masks', action='store_true', help='use masks before application of the loss')
        parser.add('--ssm_calc_grad', action='store_true', help='if True, the loss is differentiable')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.calc_grad = args.ssm_calc_grad
        self.use_masks = args.ssm_use_masks

        self.loss = SSIM()

    def forward(self, data_dict, losses_dict):
        real_imgs = data_dict['target_imgs']
        fake_imgs = data_dict['pred_target_imgs']
        
        b, t, c, h, w = real_imgs.shape
        real_imgs = real_imgs.view(-1, c, h, w)
        fake_imgs = fake_imgs.view(-1, c, h, w)

        if self.use_masks:
            real_segs = data_dict['real_segs'].view(b*t, -1, h, w)

            real_imgs = real_imgs * real_segs
            fake_imgs = fake_imgs * real_segs

        # Calculate the loss
        if self.calc_grad:
            loss = self.loss(fake_imgs, real_imgs)
        else:
            with torch.no_grad():
                loss = self.loss(fake_imgs.detach(), real_imgs)

        losses_dict['G_SSIM'] = loss.mean()

        return losses_dict


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)