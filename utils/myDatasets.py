from typing import Any, Callable, Optional, Tuple
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

''' Wrapper class for loading pytorch dataset to memory '''
class Dataset2Memory(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, dataset: torch.utils.data.Dataset, train: bool, args: ArgumentParser):
        self.images, self.labels = self.load_data_to_memory(dataset, dataset_name)

        if train and args.augmentation:
            self.augments = transforms.Compose([
                # transforms.ColorJitter(brightness=0),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.augments = None

        self.args = args

        # print("Custom dataset loaded to memory")
        # print("[Custom] shape:", self.images.shape)
        # print("[Custom] labels:", len(self.labels))

    def load_data_to_memory(self, dataset: torch.utils.data.Dataset, dataset_name):
        images = []
        labels = []
        for i in tqdm(range(len(dataset)), desc=f"Loading {dataset_name} to memory"):
            img, target = dataset.__getitem__(i)
            images.append(img)
            labels.append(target)
        return torch.stack(images), labels

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.images[index], self.labels[index]
        if self.augments is not None:
            img = self.augments(img)
        return img, target