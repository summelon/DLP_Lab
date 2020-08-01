import os
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
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


def transform_func(mode):
    if mode == 'train':
        return transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])


class RetinopathyDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(root, mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    # TODO Check if label is transformed to long tensor
    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'

           step2. Get the ground truth label from self.label

           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping,
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints.

                  In the testing phase, if you have a normalization process during the training phase, you only need
                  to normalize the data.

                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]

            step4. Return processed image and label
        """
        img = Image.open(os.path.join(self.root, 'images',
                                      self.img_name[index]+'.jpeg'))
        img = transform_func(self.mode)(img)
        lbl = torch.tensor(self.label[index], dtype=torch.long)

        return img, lbl

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


if __name__ == '__main__':
    dataset = RetinopathyDataset('./data/', 'val')
    dataset.check_image()
