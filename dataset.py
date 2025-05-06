"""Segmentation dataset loader for Mask R-CNN training."""

import os
import cv2
import torch
import numpy as np
import skimage.io as sio
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Custom dataset class for instance segmentation with mask, bbox, and label annotations.
    """

    def __init__(self, root_dir, img_dirs, transforms):
        """
        Args:
            root_dir (str): Path to dataset root.
            img_dirs (List[str]): List of image folder names.
            transforms (callable): Albumentations transform.
        """
        self.root_dir = root_dir
        self.img_dirs = sorted(img_dirs)  # 排序以避免每次順序不同
        self.transforms = transforms

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, index):
        img_dir = self.img_dirs[index]
        img_path = os.path.join(self.root_dir, img_dir, 'image.tif')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks, bboxes, labels = [], [], []

        for class_idx in range(1, 5):
            mask_path = os.path.join(
                self.root_dir, img_dir, f"class{class_idx}.tif")
            if not os.path.exists(mask_path):
                continue

            mask = sio.imread(mask_path)
            for label_id in np.unique(mask):
                if label_id == 0:
                    continue

                component_mask = (mask == label_id).astype(np.uint8)
                pos = np.where(component_mask)

                if pos[0].size == 0 or pos[1].size == 0:
                    continue

                x_min, x_max = np.min(pos[1]), np.max(pos[1])
                y_min, y_max = np.min(pos[0]), np.max(pos[0])

                if (x_max - x_min) < 1 or (y_max - y_min) < 1:
                    continue

                masks.append(component_mask)
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_idx)

        if self.transforms:
            augmented = self.transforms(
                image=image, bboxes=bboxes, masks=masks, labels=labels)
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            masks = augmented["masks"]
            labels = augmented["labels"]

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.stack([
            torch.from_numpy(m).to(
                torch.uint8) if isinstance(
                m,
                np.ndarray) else m.clone().detach().to(
                torch.uint8)
            for m in masks
        ], dim=0)

        target = {
            'boxes': bboxes,
            'labels': labels,
            'masks': masks
        }

        return image, target
