import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from importlib import import_module
import cv2



def get_dataloader(args, phase):
    dataset = import_module(f'datasets.{args.dataloader_name}').DatasetWrapper(args, phase)
    if phase == 'train': 
    	args.train_size = len(dataset)
    return DataLoader(dataset, 
        batch_size=args.batch_size // args.world_size, 
        sampler=DistributedSampler(dataset, args.world_size, args.rank, shuffle=False), # shuffling is done inside the dataset
        num_workers=args.num_workers_per_process,
        drop_last=True)

# Required to draw a stickman for ArcSoft keypoints
def merge_parts(part_even, part_odd):
    output = []
    
    for i in range(len(part_even) + len(part_odd)):
        if i % 2:
            output.append(part_odd[i // 2])
        else:
            output.append(part_even[i // 2])

    return output

# Function for stickman and facemasks drawing
def draw_stickmen(args, poses):
    ### Define drawing options ###
    if not '2d' in args.folder_postfix and not '3d' in args.folder_postfix:
        # Arcsoft keypoints
        edges_parts  = [
            merge_parts(range(0, 19), range(103, 121)), # face
            list(range(19, 29)), list(range(29, 39)), # eyebrows
            merge_parts(range(39, 51), range(121, 133)), list(range(165, 181)), [101, 101], # right eye
            merge_parts(range(51, 63), range(133, 145)), list(range(181, 197)), [102, 102], # left eye
            list(range(63, 75)), list(range(97, 101)), # nose
            merge_parts(range(75, 88), range(145, 157)), merge_parts(range(157, 165), range(88, 95))] # lips

        closed_parts = [
            False, 
            True, True, 
            True, True, False, 
            True, True, False, 
            False, False, 
            True, True]

        colors_parts = [
            (  255,  255,  255), 
            (  255,    0,    0), (    0,  255,    0),
            (    0,    0,  255), (    0,    0,  255), (    0,    0,  255),
            (  255,    0,  255), (  255,    0,  255), (  255,    0,  255),
            (    0,  255,  255), (    0,  255,  255),
            (  255,  255,    0), (  255,  255,    0)]

    else:
        edges_parts  = [
            list(range( 0, 17)), # face
            list(range(17, 22)), list(range(22, 27)), # eyebrows (right left)
            list(range(27, 31)) + [30, 33], list(range(31, 36)), # nose
            list(range(36, 42)), list(range(42, 48)), # right eye, left eye
            list(range(48, 60)), list(range(60, 68))] # lips

        closed_parts = [
            False, False, False, False, False, True, True, True, True]

        colors_parts = [
            (  255,  255,  255), 
            (  255,    0,    0), (    0,  255,    0),
            (    0,    0,  255), (    0,    0,  255), 
            (  255,    0,  255), (    0,  255,  255),
            (  255,  255,    0), (  255,  255,    0)]

    ### Start drawing ###
    stickmen = []

    for pose in poses:
        if isinstance(pose, torch.Tensor):
            # Apply conversion to numpy, asssuming the range to be in [-1, 1]
            xy = (pose.view(-1, 2).cpu().numpy() + 1) / 2 * args.image_size
        
        else:
            # Assuming the range to be [0, 1]
            xy = pose[:, :2] * self.args.image_size

        xy = xy[None, :, None].astype(np.int32)

        stickman = np.ones((args.image_size, args.image_size, 3), np.uint8)

        for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):
            stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=args.stickmen_thickness)

        stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
        stickmen.append(stickman)

    stickmen = torch.stack(stickmen)
    stickmen = (stickmen - 0.5) * 2. 

    return stickmen

# Flip vector poses via x axis
def flip_poses(args, keypoints, size):
    if not '2d' in args.folder_postfix and not '3d' in args.folder_postfix:
        # Arcsoft keypoints
        edges_parts  = [
            merge_parts(range(0, 19), range(103, 121)), # face
            list(range(19, 29)), list(range(29, 39)), # eyebrows
            merge_parts(range(39, 51), range(121, 133)), list(range(165, 181)), [101, 101], # right eye
            merge_parts(range(51, 63), range(133, 145)), list(range(181, 197)), [102, 102], # left eye
            list(range(63, 75)), list(range(97, 101)), # nose
            merge_parts(range(75, 88), range(145, 157)), merge_parts(range(157, 165), range(88, 95))] # lip

    else:
        edges_parts  = [
            list(range( 0, 17)), # face
            list(range(17, 22)), list(range(22, 27)), # eyebrows (right left)
            list(range(27, 31)) + [30, 33], list(range(31, 36)), # nose
            list(range(36, 42)), list(range(42, 48)), # right eye, left eye
            list(range(48, 60)), list(range(60, 68))] # lips


    keypoints[:, 0] = size - keypoints[:, 0]

    # Swap left and right face parts
    if not '2d' in args.folder_postfix and not '3d' in args.folder_postfix:
        l_parts  = edges_parts[1] + edges_parts[3] + edges_parts[4] + edges_parts[5][:1]
        r_parts = edges_parts[2] + edges_parts[6] + edges_parts[7] + edges_parts[8][:1]

    else:
        l_parts = edges_parts[2] + edges_parts[6]
        r_parts = edges_parts[1] + edges_parts[5]

    keypoints[l_parts + r_parts] = keypoints[r_parts + l_parts]

    return keypoints