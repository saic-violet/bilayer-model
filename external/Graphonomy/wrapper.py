import sys
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2

from .networks import deeplab_xception_transfer, graph



class SegmentationWrapper(nn.Module):
    def __init__(self, args):
        super(SegmentationWrapper, self).__init__()
        self.use_gpus = args.num_gpus > 0

        self.net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
            n_classes=20, hidden_layers=128, source_classes=7)
        
        x = torch.load(f'{args.project_dir}/pretrained_weights/graphonomy/pretrained_model.pth')
        self.net.load_source_model(x)

        if self.use_gpus:
            self.net.cuda()

        self.net.eval()

        # transforms
        self.rgb2bgr = transforms.Lambda(lambda x:x[:, [2,1,0],...])

        # adj
        adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
        self.adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

        adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
        self.adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

        cihp_adj = graph.preprocess_adj(graph.cihp_graph)
        adj3_ = Variable(torch.from_numpy(cihp_adj).float())
        self.adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

        # Erosion kernel
        SIZE = 5

        grid = np.meshgrid(np.arange(-SIZE//2+1, SIZE//2+1), np.arange(-SIZE//2+1, SIZE//2+1))[:2]
        self.kernel = (grid[0]**2 + grid[1]**2 < (SIZE / 2.)**2).astype('uint8')

    def forward(self, imgs):
        b, t = imgs.shape[:2]
        imgs = imgs.view(b*t, *imgs.shape[2:])

        inputs = self.rgb2bgr(imgs)
        inputs = Variable(inputs, requires_grad=False)

        if self.use_gpus:
            inputs = inputs.cuda()

        outputs = []
        with torch.no_grad():
            for input in inputs.split(1, 0):
                outputs.append(self.net.forward(input, self.adj1_test, self.adj3_test, self.adj2_test))
        outputs = torch.cat(outputs, 0)

        outputs = F.softmax(outputs, 1)

        segs = 1 - outputs[:, [0]] # probabilities for FG

        # Erosion
        segs_eroded = []
        for seg in segs.split(1, 0):
            seg = cv2.erode(seg[0, 0].cpu().numpy(), self.kernel, iterations=1)
            segs_eroded.append(torch.from_numpy(seg))
        segs = torch.stack(segs_eroded)[:, None].to(imgs.device)

        return segs