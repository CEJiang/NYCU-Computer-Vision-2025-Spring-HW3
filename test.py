"""
Test module for model inference with Test-Time Augmentation (TTA).
"""

import os
import json
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_utils
from torchvision.ops import nms
from tqdm import tqdm


class TestDataset(Dataset):
    """Dataset class for loading test images and metadata."""

    def __init__(self, img_dir, json_file):
        self.img_dir = img_dir
        with open(json_file, 'r', encoding='utf-8') as json_fp:
            self.img_info = json.load(json_fp)
        self.img_info.sort(key=lambda x: x["id"])

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        """Load image and metadata for a given index."""
        info = self.img_info[index]
        file_name = info["file_name"]
        image_id = info["id"]
        height = info["height"]
        width = info["width"]

        image_path = os.path.join(self.img_dir, file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, image_id, height, width


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return tuple(zip(*batch))


def load_data(data_path, batch_size):
    """Load test dataset as DataLoader."""
    dataset = TestDataset(
        img_dir=os.path.join(data_path, "test_release"),
        json_file=os.path.join(data_path, "test_image_name_to_ids.json")
    )
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, collate_fn=collate_fn)


def apply_nms_and_format(preds, image_id, height, width, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) and format results in COCO style."""
    if len(preds) == 0:
        return []

    boxes = torch.tensor([p["bbox"] for p in preds], dtype=torch.float32)
    scores = torch.tensor([p["score"] for p in preds], dtype=torch.float32)
    keep = nms(boxes, scores, iou_threshold)

    results = []
    for idx in keep:
        pred = preds[idx]
        x_min, y_min, x_max, y_max = pred["bbox"]
        mask = pred["mask"].astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')

        results.append({
            "image_id": int(image_id),
            "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
            "score": float(pred["score"]),
            "category_id": int(pred["label"]),
            "segmentation": {
                "size": [height, width],
                "counts": rle['counts']
            }
        })

    return results


@torch.no_grad()
def test_model(device, model, args):
    """Run inference with TTA and save predictions to file."""
    model_path = os.path.join(args.saved_path, "best_model_13.pth")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    test_loader = load_data(args.data_path, args.batch_size)
    all_results = []

    tta_list = [
        ("identity", A.Compose([]), A.Compose([])),
        ("hflip", A.Compose([A.HorizontalFlip(p=1.0)]),
         A.Compose([A.HorizontalFlip(p=1.0)])),
        ("vflip", A.Compose([A.VerticalFlip(p=1.0)]),
         A.Compose([A.VerticalFlip(p=1.0)])),
        ("rotate90", A.Compose([A.Rotate(limit=(90, 90), p=1.0)]), A.Compose(
            [A.Rotate(limit=(-90, -90), p=1.0)])),
        ("rotate180", A.Compose([A.Rotate(limit=(180, 180), p=1.0)]), A.Compose(
            [A.Rotate(limit=(180, 180), p=1.0)])),
        ("bright_contrast", A.Compose([A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=1.0)]), A.Compose([])),
        ("hue",
         A.Compose([A.HueSaturationValue(hue_shift_limit=10,
                                         sat_shift_limit=15,
                                         val_shift_limit=10,
                                         p=1.0)]),
         A.Compose([])),
        ("blur", A.Compose(
            [A.GaussianBlur(blur_limit=(3, 5), p=1.0)]), A.Compose([])),
        ("transpose", A.Compose([A.Transpose(p=1.0)]),
         A.Compose([A.Transpose(p=1.0)])),
    ]

    norm_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    for images, image_ids, heights, widths in tqdm(
            test_loader, desc="Testing"):
        for image, image_id, height, width in zip(
                images, image_ids, heights, widths):
            merged_preds = []
            for _, tta_aug, tta_reverse in tta_list:
                img_aug = tta_aug(image=image)["image"]
                img_tensor = norm_transform(
                    image=img_aug)["image"].unsqueeze(0).to(device)
                output = model(img_tensor)[0]

                boxes = output['boxes'].cpu().numpy()
                masks = output['masks'].squeeze(1).cpu().numpy()

                for i, box in enumerate(boxes):
                    x_min, y_min, x_max, y_max = box
                    mask = masks[i] > 0.5
                    mask_rev = tta_reverse(image=mask.astype(np.float32))[
                        "image"] > 0.5

                    box_mask = np.zeros_like(mask, dtype=np.uint8)
                    box_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
                    rev_box = tta_reverse(
                        image=box_mask.astype(
                            np.float32))["image"]

                    y_coords, x_coords = np.where(rev_box > 0.5)
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        x_min, x_max = x_coords.min(), x_coords.max()
                        y_min, y_max = y_coords.min(), y_coords.max()
                        merged_preds.append({
                            "bbox": [x_min, y_min, x_max, y_max],
                            "score": float(output['scores'][i]),
                            "label": int(output['labels'][i]),
                            "mask": mask_rev
                        })

            final_preds = apply_nms_and_format(
                merged_preds, image_id, height, width)
            all_results.extend(final_preds)

    os.makedirs(args.saved_path, exist_ok=True)
    pred_json_path = os.path.join(args.saved_path, "test-results.json")
    with open(pred_json_path, "w", encoding="utf-8") as json_fp:
        json.dump(all_results, json_fp)
    print(f"Segmentation predictions saved to {pred_json_path}")
