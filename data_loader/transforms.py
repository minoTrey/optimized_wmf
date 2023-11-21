import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""
    # initial disparity range : 0~255
    # ground truth disparity range : 0~65535

    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, sample):
        for k in sample.keys():
            if 'name' in k or 'shape' in k:
                pass
            else:
                sample[k] = np.array(sample[k])
            if isinstance(k, tuple):
                if 'img' in k[0]:
                    if self.normalize:
                        sample[k] = torch.from_numpy(sample[k].transpose(2, 0, 1)) / 255.
                    else:
                        sample[k] = torch.from_numpy(sample[k].transpose(2, 0, 1))  # [C,H,W]
                elif 'init' in k[0]:
                    sample[k] = torch.from_numpy(sample[k]).unsqueeze(0) / 255.
                elif 'gt' in k[0]:
                    sample[k] = torch.from_numpy(sample[k]).unsqueeze(0) / 256. / 255.
            else:
                if 'img' in k:
                    if self.normalize:
                        sample[k] = torch.from_numpy(sample[k].transpose(2, 0, 1)) / 255.
                    else:
                        sample[k] = torch.from_numpy(sample[k].transpose(2, 0, 1))
                elif 'init' in k:
                    sample[k] = torch.from_numpy(sample[k]).unsqueeze(0) / 255.
                elif 'gt' in k:
                    sample[k] = torch.from_numpy(sample[k]).unsqueeze(0) / 256. / 255.

        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(k, tuple):
                if 'img' in k[0]:
                    # Images have converted to tensor, with shape [C, H, W]
                    for t, m, s in zip(sample[k], self.mean, self.std):
                        t.sub_(m).div_(s)
            else:
                if 'img' in k:
                    # Images have converted to tensor, with shape [C, H, W]
                    for t, m, s in zip(sample[k], self.mean, self.std):
                        t.sub_(m).div_(s)

        return sample


class Resize(object):

    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(k, tuple):
                if 'img' in k[0]:
                    sample[k] = torchvision.transforms.Resize((self.img_height, self.img_width))(sample[k])
            else:
                if 'img' in k:
                    sample[k] = torchvision.transforms.Resize((self.img_height, self.img_width))(sample[k])

        return sample


class CenterCrop(object):
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(k, tuple):
                if 'img' in k[0]:
                    sample[k] = torchvision.transforms.CenterCrop((self.img_height, self.img_width))(sample[k])
            else:
                if 'img' in k:
                    sample[k] = torchvision.transforms.CenterCrop((self.img_height, self.img_width))(sample[k])

        return sample


class Pad(object):
    def __init__(self, top_pad, right_pad):
        self.top_pad = top_pad
        self.right_pad = right_pad

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(k, tuple):
                if 'img' in k[0]:
                    sample[k] = F.pad(sample[k], (0, self.top_pad, self.right_pad, 0))
            else:
                if 'img' in k:
                    sample[k] = F.pad(sample[k], (0, self.top_pad, self.right_pad, 0))
        return sample
