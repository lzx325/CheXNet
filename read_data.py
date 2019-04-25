# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
np.random.seed(10)


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        labels_arr = np.array(self.labels)
        n_images, n_classes = labels_arr.shape
        self.pos_counts = np.array(self.labels).sum(0)
        self.neg_counts = n_images-self.pos_counts
        self.pos_weight = n_images/self.pos_counts/2
        self.neg_weight = n_images/self.neg_counts/2
        weight = (1/self.pos_counts)/np.sum(1/self.pos_counts)
        self.class_weight = n_classes*weight
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


class ChestXrayDataSetWithAugmentation(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        # self.image_names self.labels

        aug_index = augment_index(labels)

        self.image_names = image_names
        self.labels = labels
        self.aug_index = aug_index
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        index = self.aug_index[index]

        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.aug_index)


def augment_index(labels):
    labels = np.array(labels)
    n_images, n_classes = labels.shape
    pos_counts = labels.sum(0)
    pos_counts_median = np.median(pos_counts)
    num_augment = pos_counts_median-pos_counts
    num_augment[num_augment <= 0] = 0
    num_augment = np.floor(num_augment).astype(np.int64)
    print("Use augmentation to balance dataset, augmentation level for each class: ")
    print(num_augment)
    augmented_index = list(range(n_images))

    for c in range(n_classes):
        if num_augment[c] > 0:
            pos_index = (labels[:, c] > 0).nonzero()[0]
            add_index = np.random.choice(
                pos_index, num_augment[c], replace=True)
            augmented_index.extend(list(add_index))
    return augmented_index
