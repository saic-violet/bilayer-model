import torch
from torch.utils import data
from torchvision import transforms
import glob
import pathlib
from PIL import Image
import numpy as np
import pickle as pkl
import cv2
import random
import math

from datasets import utils as ds_utils
from runners import utils as rn_utils



class DatasetWrapper(data.Dataset):
    @staticmethod
    def get_args(parser):
        # Common properties
        parser.add('--num_source_frames',     default=1, type=int,
                                              help='number of frames used for initialization of the model')

        parser.add('--num_target_frames',     default=1, type=int,
                                              help='number of frames per identity used for training')

        parser.add('--image_size',            default=256, type=int,
                                              help='output image size in the model')

        parser.add('--num_keypoints',         default=68, type=int,
                                              help='number of keypoints (depends on keypoints detector)')

        parser.add('--output_segmentation',   default='True', type=rn_utils.str2bool, choices=[True, False],
                                              help='read segmentation mask')

        parser.add('--output_stickmen',       default='True', type=rn_utils.str2bool, choices=[True, False],
                                              help='draw stickmen using keypoints')
        
        parser.add('--stickmen_thickness',    default=2, type=int, help='thickness of lines in the stickman',
                                              help='thickness of lines in the stickman')

        return parser

    def __init__(self, args, phase):
        super(DatasetWrapper, self).__init__()
        # Store options
        self.phase = phase
        self.args = args

        self.to_tensor = transforms.ToTensor()
        self.epoch = 0 if args.which_epoch == 'none' else int(args.which_epoch)

        # Data paths
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / phase

        if args.output_segmentation:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / phase

        # Video sequences list
        sequences = self.imgs_dir.glob('*/*')
        self.sequences = ['/'.join(seq.split('/')[-2:]) for seq in sequences]

        # Parameters of the sampling scheme
        self.delta = math.sqrt(5)
        self.cur_num = torch.rand(1).item()

    def __getitem__(self, index):
        # Sample source and target frames for the current sequence
        while True:
            try:
                filenames_img = list((self.imgs_dir / self.sequences[index]).glob('*/*'))
                filenames_img = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_img]

                filenames_npy = list((self.pose_dir / self.sequences[index]).glob('*/*'))
                filenames_npy = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_npy]

                filenames = list(set(filenames_img).intersection(set(filenames_npy)))

                if self.args.output_segmentation:
                    filenames_seg = list((self.segs_dir / self.sequences[index]).glob('*/*'))
                    filenames_seg = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_seg]

                    filenames = list(set(filenames).intersection(set(filenames_seg)))

                if len(filenames):
                    break
                else:
                    raise

            except:
                # Exception is raised if filenames list is empty or there was an error during read
                print('Encountered an error while reading the dataset')
                index = (index + 1) % len(self)

        filenames = sorted(filenames)

        imgs = []
        poses = []
        stickmen = []
        segs = []

        reserve_index = -1 # take this element of the sequence if loading fails
        sample_from_reserve = False

        if self.phase == 'test':
            # Sample from the beginning of the sequence
            self.cur_num = 0

        while len(imgs) < self.args.num_source_frames + self.args.num_target_frames:
            if reserve_index == len(filenames):
                raise # each element of the filenames list is unavailable for load

            # Sample a frame number
            if sample_from_reserve:
                filename = filenames[reserve_index]

            else:
                frame_num = int(round(self.cur_num * (len(filenames) - 1)))
                self.cur_num = (self.cur_num + self.delta) % 1

                filename = filenames[frame_num]

            # Read images
            img_path = pathlib.Path(self.imgs_dir) / filename.with_suffix('.jpg')
            
            try:
                img = Image.open(img_path)

                # Preprocess an image
                s = img.size[0]
                img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)

            except:
                sample_from_reserve = True
                reserve_index += 1
                continue

            imgs += [self.to_tensor(img)]

            # Read keypoints
            keypoints_path = pathlib.Path(self.pose_dir) / filename.with_suffix('.npy')
            try:
                keypoints = np.load(keypoints_path).astype('float32')
            except:
                imgs.pop(-1)

                sample_from_reserve = True
                reserve_index += 1
                continue

            keypoints = keypoints[:self.args.num_keypoints, :]
            keypoints[:, :2] /= s
            keypoints = keypoints[:, :2]

            poses += [torch.from_numpy(keypoints.reshape(-1))]

            if self.args.output_segmentation:
                seg_path = pathlib.Path(self.segs_dir) / filename.with_suffix('.png')

                try:
                    seg = Image.open(seg_path)
                    seg = seg.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
                except:
                    imgs.pop(-1)
                    poses.pop(-1)

                    sample_from_reserve = True
                    reserve_index += 1
                    continue

                segs += [self.to_tensor(seg)]

            sample_from_reserve = False

        imgs = (torch.stack(imgs)- 0.5) * 2.0

        poses = (torch.stack(poses) - 0.5) * 2.0

        if self.args.output_stickmen:
            stickmen = utils.draw_stickmen(self.args, poses)

        if self.args.output_segmentation:
            segs = torch.stack(segs)

        # Split between few-shot source and target sets
        data_dict = {}
        if self.args.num_source_frames:
            data_dict['source_imgs'] = imgs[:self.args.num_source_frames]
        data_dict['target_imgs'] = imgs[self.args.num_source_frames:]
        
        if self.args.num_source_frames:
            data_dict['source_poses'] = poses[:self.args.num_source_frames]
        data_dict['target_poses'] = poses[self.args.num_source_frames:]

        if self.args.output_stickmen:
            if self.args.num_source_frames:
                data_dict['source_stickmen'] = stickmen[:self.args.num_source_frames]
            data_dict['target_stickmen'] = stickmen[self.args.num_source_frames:]
        
        if self.args.output_segmentation:
            if self.args.num_source_frames:
                data_dict['source_segs'] = segs[:self.args.num_source_frames]
            data_dict['target_segs'] = segs[self.args.num_source_frames:]
        
        data_dict['indices'] = torch.LongTensor([index])

        return data_dict

    def __len__(self):
        return len(self.sequences)

    def shuffle(self):
        self.sequences = [self.sequences[i] for i in torch.randperm(len(self.sequences)).tolist()]