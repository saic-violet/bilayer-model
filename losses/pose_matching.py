# Third party
import torch
from torch import nn
import torch.nn.functional as F
from joblib import Parallel, delayed
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import sys
sys.path.append('/group-volume/orc_srr/violet/e.zakharov/projects/face-alignment')
import face_alignment

# This project
from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--nme_num_threads', type=int, default=8)
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.num_threads = args.nme_num_threads

        # Supported loss functions
        losses = {
            'mse': F.mse_loss,
            'l1': F.l1_loss}

        self.fa = []

        for i in range(self.num_threads):
            self.fa.append(face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda'))

        # Used to calculate a normalization factor
        self.right_eye = list(range(36, 42))
        self.left_eye = list(range(42, 48))

    def forward(self, data_dict, losses_dict):
        fake_imgs = data_dict['pred_target_imgs']
        real_imgs = data_dict['target_imgs']

        b, t = real_imgs.shape[:2]

        fake_imgs = fake_imgs.view(b*t, *fake_imgs.shape[2:])
        real_imgs = real_imgs.view(b*t, *real_imgs.shape[2:])

        losses = [self.calc_metric(fake_img, real_img, i % self.num_threads) for i, (fake_img, real_img) in enumerate(zip(fake_imgs, real_imgs))]

        losses_dict['G_PME'] = sum(losses) / len(losses)

        return losses_dict

    @torch.no_grad()
    def calc_metric(self, fake_img, real_img, worker_id):
        fake_img = (((fake_img.detach() + 1.0) / 2.0) * 255.0).cpu().numpy().astype('uint8').transpose(1, 2, 0)
        fake_keypoints = torch.from_numpy(self.fa[worker_id].get_landmarks(fake_img)[0])[:, :2]

        real_img = (((real_img.detach() + 1.0) / 2.0) * 255.0).cpu().numpy().astype('uint8').transpose(1, 2, 0)
        real_keypoints = torch.from_numpy(self.fa[worker_id].get_landmarks(real_img)[0])[:, :2]

        # Calcualte normalization factor
        d = ((real_keypoints[self.left_eye].mean(0) - real_keypoints[self.right_eye].mean(0))**2).sum()**0.5

        # Calculate the mean error
        error = torch.mean(((fake_keypoints - real_keypoints)**2).sum(1)**0.5)

        loss = error / d

        return loss
