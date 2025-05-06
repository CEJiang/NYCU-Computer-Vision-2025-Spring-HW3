"""
Training pipeline for Mask R-CNN with Swin Transformer backbone.
Handles data loading, augmentation, training, validation, and checkpointing.
"""

import os
import time
import json
import gc
import tempfile
from contextlib import redirect_stdout

import numpy as np
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import SegmentationDataset
from repeated_aug_sampler import RepeatedAugSampler
from utils import plot_loss_accuracy


def get_transforms(train=False):
    """Define data augmentations using Albumentations."""
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'])

    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.OpticalDistortion(distort_limit=0.05, p=0.2),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=bbox_params)

    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=bbox_params)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return tuple(zip(*batch))


def load_data(data_path, args):
    """Load training and validation datasets with augmentation."""
    ratio = 0.8 if args.is_valid else 1.0
    all_dirs = sorted(os.listdir(data_path))

    train_size = int(ratio * len(all_dirs))
    train_dirs = all_dirs[:train_size]
    valid_dirs = all_dirs[train_size:]

    train_dataset = SegmentationDataset(
        root_dir=data_path,
        img_dirs=train_dirs,
        transforms=get_transforms(train=True))

    valid_dataset = SegmentationDataset(
        root_dir=data_path,
        img_dirs=valid_dirs,
        transforms=get_transforms(train=False))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=RepeatedAugSampler(train_dataset, num_repeats=3),
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn)

    num_classes = 5
    return train_loader, valid_loader, num_classes


class Trainer:
    """Trainer class to handle training and validation."""

    def __init__(self, device, model, optimizer, scheduler, args):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.train_losses = []
        self.maps = []
        self.best_loss = 0.0
        self.best_map = 0.0

    def train(self, train_loader, epoch):
        """Perform one epoch of training."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            print(
                f"  ├─ Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def validate(self, valid_loader, epoch):
        """Run COCO mAP50 validation."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        coco_results = []
        gt_annotations = []
        image_shapes = []
        image_id = 0
        annotation_id = 0

        for images, targets in valid_loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]

            outputs = self.model(images)
            loss_dict = self.model(images, targets)

            if isinstance(loss_dict, dict):
                total_loss += sum(loss for loss in loss_dict.values()).item()
                count += 1

            for output, target, img in zip(outputs, targets, images):
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                masks = output['masks'].squeeze(1).cpu().numpy()

                for mask_idx, mask_data in enumerate(masks):
                    mask = (mask_data > 0.5).astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    coco_results.append({
                        'image_id': image_id,
                        'category_id': int(labels[mask_idx]),
                        'segmentation': rle,
                        'score': float(scores[mask_idx])
                    })

                gt_masks = target['masks'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()

                for mask, label in zip(gt_masks, gt_labels):
                    mask = mask.astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    gt_annotations.append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': int(label),
                        'segmentation': rle,
                        'area': float(np.sum(mask)),
                        'iscrowd': 0
                    })
                    annotation_id += 1

                height, width = img.shape[1], img.shape[2]
                image_shapes.append({
                    'id': image_id,
                    'file_name': f'{image_id}.png',
                    'height': height,
                    'width': width
                })
                image_id += 1

        avg_loss = total_loss / count if count > 0 else None
        if avg_loss is not None:
            print(f'[Epoch {epoch+1}] Val Loss: {avg_loss:.4f}')

        if not coco_results or not gt_annotations:
            print(
                f"[Epoch {epoch+1}] No valid predictions for mask evaluation.")
            return avg_loss, None

        coco_gt = {
            "images": image_shapes,
            "annotations": gt_annotations,
            "categories": [{"id": i, "name": str(i)} for i in range(1, 5)]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file, \
                tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as dt_file:
            json.dump(coco_gt, gt_file)
            json.dump(coco_results, dt_file)
            gt_file.flush()
            dt_file.flush()

        coco_gt = COCO(gt_file.name)
        coco_dt = coco_gt.loadRes(dt_file.name)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.evaluate()
        coco_eval.accumulate()

        log_file = os.path.join(self.args.saved_path, "map_log.txt")
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(f"\n[Epoch {epoch + 1}] Mask Evaluation\n")
            with redirect_stdout(file):
                coco_eval.summarize()

        map50 = coco_eval.stats[0]
        print(f"[Epoch {epoch+1}] Mask mAP50 = {map50:.4f}")
        return avg_loss, map50

    def save_model(self, epoch, is_best=False):
        """Save the current training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_map': self.best_map,
            'train_losses': self.train_losses,
            'maps': self.maps
        }
        torch.save(
            checkpoint,
            os.path.join(
                self.args.saved_path,
                'latest_checkpoint.pth'))

        if is_best:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.args.saved_path,
                    f"best_model_{epoch}.pth"))
            print("Best model is saved.")

    def load_model(self):
        """Load a saved checkpoint."""
        path = os.path.join(self.args.saved_path, 'latest_checkpoint.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.best_map = checkpoint['best_map']
            self.train_losses = checkpoint['train_losses']
            self.maps = checkpoint['maps']
            print(
                f"Resumed from epoch {checkpoint['epoch'] + 1} | Best mAP: {self.best_map:.4f}")
            return checkpoint['epoch'] + 1
        return 0


def train_model(device, model, optimizer, scheduler,
                train_loader, valid_loader, args):
    """Training entry point."""
    trainer = Trainer(device, model, optimizer, scheduler, args)
    start_epoch = trainer.load_model()
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        train_loss = trainer.train(train_loader, epoch)
        trainer.train_losses.append(train_loss)

        if args.is_valid:
            _, map50 = trainer.validate(valid_loader, epoch)
            trainer.maps.append(map50)
            if map50 is not None and map50 >= trainer.best_map:
                trainer.best_map = map50
                trainer.save_model(epoch, is_best=True)
            else:
                trainer.save_model(epoch)
        else:
            trainer.save_model(epoch)

        gc.collect()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1} time: {time.time() - start:.2f} sec")

    plot_loss_accuracy(train_losses=trainer.train_losses, val_losses=[])


def validate_model(device, model, valid_loader, args):
    """Optional validation-only entry point."""
    raise NotImplementedError
