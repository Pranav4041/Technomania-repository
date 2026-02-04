# script/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

NUM_CLASSES = 10
IMG_SIZE = 256   # 🔥 important for stability


class SegDataset(Dataset):

    def __init__(self, img_dir, mask_dir, train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train = train

        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 🔥 resize (huge stability boost)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # ======================
        # AUGMENTATIONS
        # ======================
        if self.train:

            if np.random.rand() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            if np.random.rand() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)

            if np.random.rand() > 0.5:
                k = np.random.choice([0, 1, 2, 3])
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)

            if np.random.rand() > 0.5:
                image = cv2.GaussianBlur(image, (5, 5), 0)

        image = image.astype(np.float32) / 255.0

        # 🔥 mask safety
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return image, mask
