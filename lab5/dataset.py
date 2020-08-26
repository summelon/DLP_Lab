import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms


def trans_func(mode: str, img_size: int) -> callable:
    # resize_size = (int(img_size*1.15), int(img_size*1.15))
    if mode == "train":
        return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                # transforms.Resize(resize_size),
                # transforms.CenterCrop((img_size, img_size)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation((-30, 30)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    else:
        return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])


def gen_labels(num_cls: int, batch_size: int) -> torch.Tensor:
    labels = list()
    for i in range(batch_size):
        length = np.random.randint(1, 4, 1)
        label = np.random.choice(range(num_cls), length, replace=False)
        oh_label = torch.nn.functional.one_hot(torch.tensor(label), num_cls)
        labels.append(oh_label.sum(0).view(1, -1))
    return torch.cat(labels, dim=0)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, img_size: int, mode: str = 'train'):
        super(SyntheticDataset, self).__init__()

        self.img_size = img_size
        self.mode = mode
        label_df = pd.read_json('./dataset/objects.json', lines=True)
        self.label_map = label_df.iloc[0].to_dict()
        self.img_list, self.label_list = self.read_json()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        if self.mode == 'train':
            img_path = os.path.join('./dataset/images', img)
            # Convert RGBA to RGB
            img = Image.open(img_path).convert('RGB')
            img = trans_func(self.mode, self.img_size)(img)
        label = self.sps2oh(self.label_list[idx])
        return img, label

    def read_json(self) -> (list, list):
        if self.mode == 'train':
            train_data = pd.read_json('./dataset/train.json', lines=True)
            imgs = train_data.columns.to_list()
            labels = train_data.loc[0].to_list()
            return imgs, labels
        elif self.mode == 'val':
            val_data = pd.read_json('./dataset/test.json', lines=True)
            labels = val_data.iloc[0].to_list()
            imgs = val_data.columns.to_list()
            return imgs, labels
        else:
            raise ValueError(f'Input mode {self.mode} is wrong!')

    def sps2oh(self, label: list) -> list:
        oh_label = torch.tensor([self.label_map[lbl] for lbl in label])
        return torch.nn.functional.one_hot(
                oh_label, len(self.label_map)).sum(dim=0)
