from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from .mypath_atr import Path
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VOCSegmentation(Dataset):
    """
    ATR dataset
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('atr'),
                 split='train',
                 transform=None,
                 flip=False,
                 ):
        """
        :param base_dir: path to ATR dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super(VOCSegmentation).__init__()
        self._flip_flag = flip

        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        self._flip_dir = os.path.join(self._base_dir,'SegmentationClassAug_rev')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'list')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.flip_categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '_id.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line+'.jpg' )
                _cat = os.path.join(self._cat_dir, line +'.png')
                _flip = os.path.join(self._flip_dir,line + '.png')
                # print(self._image_dir,_image)
                assert os.path.isfile(_image)
                # print(_cat)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_flip)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.flip_categories.append(_flip)


        assert (len(self.images) == len(self.categories))
        assert len(self.flip_categories) == len(self.categories)

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        """
        Returns the length of the image.

        Args:
            self: (todo): write your description
        """
        return len(self.images)


    def __getitem__(self, index):
        """
        Get the index of an item

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        """
        Make image coordinates

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')  # return is RGB pic
        if self._flip_flag:
            if random.random() < 0.5:
                _target = Image.open(self.flip_categories[index])
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                _target = Image.open(self.categories[index])
        else:
            _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        """
        Return a string representation of the string.

        Args:
            self: (todo): write your description
        """
        return 'ATR(split=' + str(self.split) + ')'



