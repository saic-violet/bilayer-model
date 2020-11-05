import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps
from torchvision import transforms

class RandomCrop(object):
    def __init__(self, size, padding=0):
        """
        Initialize the size.

        Args:
            self: (todo): write your description
            size: (int): write your description
            padding: (str): write your description
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        """
        Call the image.

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img, mask = sample['image'], sample['label']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            return {'image': img,
                    'label': mask}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}

class RandomCrop_new(object):
    def __init__(self, size, padding=0):
        """
        Initialize the size.

        Args:
            self: (todo): write your description
            size: (int): write your description
            padding: (str): write your description
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        """
        Call this image.

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img, mask = sample['image'], sample['label']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}

        new_img = Image.new('RGB',(tw,th),'black')  # size is w x h; and 'white' is 255
        new_mask = Image.new('L',(tw,th),'white')  # same above

        # if w > tw or h > th
        x1 = y1 = 0
        if w > tw:
            x1 = random.randint(0,w - tw)
        if h > th:
            y1 = random.randint(0,h - th)
        # crop
        img = img.crop((x1,y1, x1 + tw, y1 + th))
        mask = mask.crop((x1,y1, x1 + tw, y1 + th))
        new_img.paste(img,(0,0))
        new_mask.paste(mask,(0,0))

        # x1 = random.randint(0, w - tw)
        # y1 = random.randint(0, h - th)
        # img = img.crop((x1, y1, x1 + tw, y1 + th))
        # mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': new_img,
                'label': new_mask}

class Paste(object):
    def __init__(self, size,):
        """
        Initialize a number of bytes.

        Args:
            self: (todo): write your description
            size: (int): write your description
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w

    def __call__(self, sample):
        """
        Call the function to the image.

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img, mask = sample['image'], sample['label']

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        assert (w <=tw) and (h <= th)
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}

        new_img = Image.new('RGB',(tw,th),'black')  # size is w x h; and 'white' is 255
        new_mask = Image.new('L',(tw,th),'white')  # same above

        new_img.paste(img,(0,0))
        new_mask.paste(mask,(0,0))

        return {'image': new_img,
                'label': new_mask}

class CenterCrop(object):
    def __init__(self, size):
        """
        Initialize a number of bytes.

        Args:
            self: (todo): write your description
            size: (int): write your description
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Call the image as a sample

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        """
        Call the image

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class HorizontalFlip(object):
    def __call__(self, sample):
        """
        Transpose the sample

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class HorizontalFlip_only_img(object):
    def __call__(self, sample):
        """
        Call the image

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip_cihp(object):
    def __call__(self, sample):
        """
        Generate random sample

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # mask = Image.open()

        return {'image': img,
                'label': mask}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        """
        Initialize the next instance.

        Args:
            self: (todo): write your description
            mean: (float): write your description
            std: (array): write your description
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Calculate function.

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class Normalize_255(object):
    """Normalize a tensor image with mean and standard deviation. tf use 255.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(123.15, 115.90, 103.06), std=(1., 1., 1.)):
        """
        Initialize a new population.

        Args:
            self: (todo): write your description
            mean: (float): write your description
            std: (array): write your description
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Call the call

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        # img = 255.0
        img -= self.mean
        img /= self.std
        img = img
        img = img[[0,3,2,1],...]
        return {'image': img,
                'label': mask}

class Normalize_xception_tf(object):
    # def __init__(self):
    #     self.rgb2bgr =

    def __call__(self, sample):
        """
        Call the call

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img = (img*2.0)/255.0 - 1
        # print(img.shape)
        # img = img[[0,3,2,1],...]
        return {'image': img,
                'label': mask}

class Normalize_xception_tf_only_img(object):
    # def __init__(self):
    #     self.rgb2bgr =

    def __call__(self, sample):
        """
        Call the call of the sample

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = np.array(sample['image']).astype(np.float32)
        # mask = np.array(sample['label']).astype(np.float32)
        img = (img*2.0)/255.0 - 1
        # print(img.shape)
        # img = img[[0,3,2,1],...]
        return {'image': img,
                'label': sample['label']}

class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        """
        Initialize the internal state.

        Args:
            self: (todo): write your description
            mean: (float): write your description
        """
        self.mean = mean

    def __call__(self, sample):
        """
        Call the call

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img -= self.mean
        img /= 255.0

        return {'image': img,
                'label': mask}

class ToTensor_(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        """
        Initialize the rgb rgb.

        Args:
            self: (todo): write your description
        """
        self.rgb2bgr = transforms.Lambda(lambda x:x[[2,1,0],...])

    def __call__(self, sample):
        """
        Call the call.

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
        # mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        img = self.rgb2bgr(img)
        mask = torch.from_numpy(mask).float()


        return {'image': img,
                'label': mask}

class ToTensor_only_img(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        """
        Initialize the rgb rgb.

        Args:
            self: (todo): write your description
        """
        self.rgb2bgr = transforms.Lambda(lambda x:x[[2,1,0],...])

    def __call__(self, sample):
        """
        Call the call

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        # mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
        # mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        img = self.rgb2bgr(img)
        # mask = torch.from_numpy(mask).float()


        return {'image': img,
                'label': sample['label']}

class FixedResize(object):
    def __init__(self, size):
        """
        Initialize the size.

        Args:
            self: (todo): write your description
            size: (int): write your description
        """
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        """
        Resize the image

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class Keep_origin_size_Resize(object):
    def __init__(self, max_size, scale=1.0):
        """
        Initialize the image.

        Args:
            self: (todo): write your description
            max_size: (int): write your description
            scale: (float): write your description
        """
        self.size = tuple(reversed(max_size))  # size: (h, w)
        self.scale = scale
        self.paste = Paste(int(max_size[0]*scale))

    def __call__(self, sample):
        """
        Call this image s output.

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size
        h, w = self.size
        h = int(h*self.scale)
        w = int(w*self.scale)
        img = img.resize((h, w), Image.BILINEAR)
        mask = mask.resize((h, w), Image.NEAREST)

        return self.paste({'image': img,
                'label': mask})

class Scale(object):
    def __init__(self, size):
        """
        Initialize a number of bytes.

        Args:
            self: (todo): write your description
            size: (int): write your description
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Call the image

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class Scale_(object):
    def __init__(self, scale):
        """
        Initialize the scale.

        Args:
            self: (todo): write your description
            scale: (float): write your description
        """
        self.scale = scale

    def __call__(self, sample):
        """
        Call the function

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        ow = int(w*self.scale)
        oh = int(h*self.scale)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class Scale_only_img(object):
    def __init__(self, scale):
        """
        Initialize the scale.

        Args:
            self: (todo): write your description
            scale: (float): write your description
        """
        self.scale = scale

    def __call__(self, sample):
        """
        Generate the image

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        # assert img.size == mask.size
        w, h = img.size
        ow = int(w*self.scale)
        oh = int(h*self.scale)
        img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomSizedCrop(object):
    def __init__(self, size):
        """
        Initialize the size.

        Args:
            self: (todo): write your description
            size: (int): write your description
        """
        self.size = size

    def __call__(self, sample):
        """
        Generate a sample.

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'label': mask}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample

class RandomRotate(object):
    def __init__(self, degree):
        """
        Initialize a degree.

        Args:
            self: (todo): write your description
            degree: (int): write your description
        """
        self.degree = degree

    def __call__(self, sample):
        """
        Generate a random sample

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomSized_new(object):
    '''what we use is this class to aug'''
    def __init__(self, size,scale1=0.5,scale2=2):
        """
        Initialize the image.

        Args:
            self: (todo): write your description
            size: (int): write your description
            scale1: (float): write your description
            scale2: (float): write your description
        """
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop_new(self.size)
        self.small_scale = scale1
        self.big_scale = scale2

    def __call__(self, sample):
        """
        Randomly sample

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        w = int(random.uniform(self.small_scale, self.big_scale) * img.size[0])
        h = int(random.uniform(self.small_scale, self.big_scale) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample = {'image': img, 'label': mask}
        # finish resize
        return self.crop(sample)
# class Random

class RandomScale(object):
    def __init__(self, limit):
        """
        Initialize a new limit.

        Args:
            self: (todo): write your description
            limit: (int): write your description
        """
        self.limit = limit

    def __call__(self, sample):
        """
        Call the sample

        Args:
            self: (todo): write your description
            sample: (int): write your description
        """
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return {'image': img, 'label': mask}