import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
import os

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0).copy()
            label = np.flip(label, 0).copy()
        if random.random() < 0.5:
            image = np.flip(image, 1).copy()
            label = np.flip(label, 1).copy()
        if random.random() < 0.5:
            image = np.flip(image, 2).copy()
            label = np.flip(label, 2).copy()

        return {'image': image, 'label': label, 'idh': sample['idh'], 'grade': sample['grade']}


class Crop_val(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        # 获取包含肿瘤区域的中心
        tumor_region = np.where(np.isin(label, [1, 2, 4]))
        if len(tumor_region[0]) == 0:  # 如果没有肿瘤区域，返回原始样本
            return sample

        # 计算肿瘤区域的中心点
        center_h = int(np.mean(tumor_region[0]))
        center_w = int(np.mean(tumor_region[1]))
        center_d = int(np.mean(tumor_region[2]))


        H_max, W_max, D_max = image.shape[:3]

        # 确定裁剪区域的起始点
        H_start = max(0, center_h - 64)
        W_start = max(0, center_w - 64)
        D_start = max(0, center_d - 64)

        # 确保裁剪区域在图像边界内
        H_start = min(H_start, H_max - 128)
        W_start = min(W_start, W_max - 128)
        D_start = min(D_start, D_max - 128)

        # 防止起始点为负
        H_start = max(0, H_start)
        W_start = max(0, W_start)
        D_start = max(0, D_start)

        image_crop = image[H_start:H_start + 128, W_start:W_start + 128, D_start:D_start + 128, ...]
        label_crop = label[H_start:H_start + 128, W_start:W_start + 128, D_start:D_start + 128]

        pad_h = max(0, 128 - image_crop.shape[0])
        pad_w = max(0, 128 - image_crop.shape[1])
        pad_d = max(0, 128 - image_crop.shape[2])

        image_crop = np.pad(image_crop, ((0, pad_h), (0, pad_w), (0, pad_d), (0, 0)), mode='constant',
                            constant_values=0)
        label_crop = np.pad(label_crop, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant', constant_values=0)

        return {'image': image_crop, 'label': label_crop, 'idh': sample['idh'], 'grade': sample['grade']}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        H_max, W_max, D_max = image.shape[:3]

        # 尝试多次裁剪，直到裁剪区域包含非零标签
        for _ in range(100):  # 最多尝试10次
            H = random.randint(0, max(0, H_max - 128))
            W = random.randint(0, max(0, W_max - 128))
            D = random.randint(0, max(0, D_max - 128))

            image_crop = image[H:H + 128, W:W + 128, D:D + 128, ...]
            label_crop = label[H:H + 128, W:W + 128, D:D + 128]

            # 检查裁剪区域是否包含非零标签
            if np.any(label_crop != 0):
                break

        # 如果经过多次尝试仍然没有非零标签，可以选择返回原始样本或其他策略
        else:
            return sample

        pad_h = max(0, 128 - image_crop.shape[0])
        pad_w = max(0, 128 - image_crop.shape[1])
        pad_d = max(0, 128 - image_crop.shape[2])

        image_crop = np.pad(image_crop, ((0, pad_h), (0, pad_w), (0, pad_d), (0, 0)), mode='constant',
                            constant_values=0)
        label_crop = np.pad(label_crop, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant', constant_values=0)

        return {'image': image_crop, 'label': label_crop, 'idh': sample['idh'], 'grade': sample['grade']}

class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label, 'idh': sample['idh'], 'grade': sample['grade']}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)
        return {'image': image, 'label': label,'idh':sample['idh'], 'grade':sample['grade']}



class guiyihua(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        mean = np.mean(image, axis=(0, 1, 2))
        std = np.std(image, axis=(0, 1, 2))

        # 对每个像素点进行归一化
        for i in range(4):
            if std[i] == 0:
                print(f"Warning: Standard deviation for channel {i} is zero. Using mean subtraction only.")
                image[:, :, :, i] = image[:, :, :, i] - mean[i]
            else:
                image[:, :, :, i] = (image[:, :, :, i] - mean[i]) / std[i]

        return {'image': image, 'label': label, 'idh': sample['idh'], 'grade': sample['grade']}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        image = torch.from_numpy(image).float()
        return {'image': image, 'label': label,'idh':sample['idh'], 'grade':sample['grade']}


def transform(sample):
    trans = transforms.Compose([
        Random_Crop(),
        guiyihua(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])
    return trans(sample)

def transform_valid(sample):
    trans = transforms.Compose([
        Crop_val(),
        guiyihua(),
        ToTensor()
    ])

    return trans(sample)


class BraTS(Dataset):
    def __init__(self, list_file, root='', mode='train'):
        self.lines = []
        paths, names, targets = [], [],[]
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':

            image, label, idh,grade= pkload(path + 'idhgrade.pkl')
            sample = {'image': image, 'label': label, 'idh':idh,'grade':grade }
            sample = transform(sample)
            return sample['image'], sample['label'], sample['idh'],sample['grade']
        elif self.mode == 'valid':
            image, label, idh, grade = pkload(path + 'idhgrade.pkl')
            sample = {'image': image, 'label': label, 'idh': idh, 'grade': grade}
            sample = transform_valid(sample)
            return sample['image'], sample['label'], sample['idh'], sample['grade']

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
