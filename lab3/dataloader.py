import os
import pandas as pd
import torch
import numpy as np
from collections import Counter
from torchvision import transforms
from PIL import Image


def getData(root, mode):
    if mode == 'train':
        img = pd.read_csv(os.path.join(root, 'train_img.csv'))
        label = pd.read_csv(os.path.join(root, 'train_label.csv'))
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv(os.path.join(root, 'test_img.csv'))
        label = pd.read_csv(os.path.join(root, 'test_label.csv'))
        return np.squeeze(img.values), np.squeeze(label.values)


def transform_func(mode, img_size):
    # import cus_aug
    # customize_aug = cus_aug.ImgAugTransform(mode)
    # return transforms.Compose([
    #             customize_aug,
    #             transforms.ToTensor()])
    if mode == 'train':
        return transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation((-30, 30)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])


class RetinopathyDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, img_size):
        self.root = root
        self.img_name, self.label = getData(root, mode)
        self.mode = mode
        self.img_size = img_size
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        img = self._open_img(index)
        img = transform_func(self.mode, self.img_size)(img)
        lbl = torch.tensor(self.label[index], dtype=torch.long)

        return img, lbl

    def _open_img(self, idx):
        return Image.open(os.path.join(self.root, 'images',
                                       self.img_name[idx]+'.jpeg'))

    def wts_sampler(self):
        if self.mode == 'train':
            counter = Counter(self.label)
            ratio = np.array([1./counter[cls] for cls in range(5)])
            wts = torch.FloatTensor([ratio[cls] for cls in self.label])
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                        weights=wts, replacement=True,
                        num_samples=len(wts))
            # return sampler
            return None
        else:
            return None

    def check_image(self):
        import matplotlib.pyplot as plt
        img_index = np.random.randint(0, len(self.label), 1)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_array = (self.__getitem__(img_index[0]))[0].numpy()
        np_array = np_array.transpose((1, 2, 0)) * std + mean
        image = np.clip(np_array, 0, 1)
        plt.imshow(image)
        plt.pause(0)

    def check_aug(self):
        customize_aug = cus_aug.ImgAugTransform()
        random_idx = np.random.randint(0, 6666, 8)
        imgs = [self._open_img(idx)[0] for idx in random_idx]
        customize_aug.check_aug(imgs)


if __name__ == '__main__':
    dataset = RetinopathyDataset('./data/', 'val')
    dataset.check_image()
    # dataset.check_aug()
