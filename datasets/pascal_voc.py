"""
datasets/pascal_voc.py

Step 3 changes vs. original
----------------------------
1. Horizontal flip augmentation (train only, off for eval).
   Box coordinates are correctly mirrored in the resized image space.
2. `build_trainval_dataset(root_dir)` — ConcatDataset of VOC2007 + VOC2012
   trainval sets → 16,551 images (PDF Sec. 3 requirement).
3. `build_test_dataset(root_dir)` — VOC2007 test set → 4,952 images.
4. `augment` flag on VOCDetectionDataset (default False; train callers pass True).
"""

import os
import random
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

# ── Class map ─────────────────────────────────────────────────────────────
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
# Labels are 1-indexed (0 = background) to match FCOSLoss / target generator.
CLASS_TO_IDX = {cls: idx + 1 for idx, cls in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class VOCDetectionDataset(Dataset):
    """Single-year Pascal VOC detection dataset.

    Args:
        root_dir  : Path to VOCdevkit/ folder.
        year      : '2007' or '2012'.
        image_set : 'train', 'val', 'trainval', or 'test'.
        min_size  : Short-edge resize target (800 keeps P6/P7 meaningful).
        max_size  : Long-edge cap to prevent OOM.
        augment   : If True, applies random horizontal flip with p=0.5.
                    Must be False for evaluation.
    """

    def __init__(
        self,
        root_dir: str,
        year: str = "2012",
        image_set: str = "trainval",
        min_size: int = 800,
        max_size: int = 1333,
        augment: bool = False,
    ):
        self.voc_root = os.path.join(root_dir, f"VOC{year}")
        self.min_size = min_size
        self.max_size = max_size
        self.augment  = augment

        imgsets_file = os.path.join(
            self.voc_root, "ImageSets", "Main", f"{image_set}.txt"
        )
        with open(imgsets_file) as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        self.img_dir = os.path.join(self.voc_root, "JPEGImages")
        self.ann_dir = os.path.join(self.voc_root, "Annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # ── 1. Load image ─────────────────────────────────────────────────
        img = Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
        orig_w, orig_h = img.size  # PIL: (W, H)

        # ── 2. Parse XML annotations ──────────────────────────────────────
        ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes, labels = [], []
        for obj in root.findall("object"):
            if int(obj.find("difficult").text) == 1:
                continue  # FCOS standard: skip difficult objects during training
            cls_name = obj.find("name").text.strip()
            if cls_name not in CLASS_TO_IDX:
                continue
            bndbox = obj.find("bndbox")
            # VOC XML is 1-indexed → convert to 0-indexed pixel coordinates
            xmin = float(bndbox.find("xmin").text) - 1
            ymin = float(bndbox.find("ymin").text) - 1
            xmax = float(bndbox.find("xmax").text) - 1
            ymax = float(bndbox.find("ymax").text) - 1
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[cls_name])

        boxes  = torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        # ── 3. Resize: short-edge = 800, long-edge ≤ 1333 ────────────────
        # 800px short edge keeps P6 (stride 64) at ~13 px and P7 (stride 128)
        # at ~7 px, which is sufficient for detection (FCOS paper setting).
        scale = self.min_size / min(orig_w, orig_h)
        if max(orig_w, orig_h) * scale > self.max_size:
            scale = self.max_size / max(orig_w, orig_h)

        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img = TF.resize(img, (new_h, new_w))

        if boxes.numel() > 0:
            boxes = boxes * scale  # scale coords into resized image space

        # ── 4. Horizontal flip augmentation (train only, p = 0.5) ─────────
        # Standard augmentation for object detection. Box transform:
        #   new_xmin = resized_W - old_xmax
        #   new_xmax = resized_W - old_xmin
        # We use new_w (resized width), not orig_w, because boxes have already
        # been scaled to the resized image space in step 3.
        if self.augment and random.random() < 0.5:
            img = TF.hflip(img)
            if boxes.numel() > 0:
                flipped = boxes.clone()
                flipped[:, 0] = new_w - boxes[:, 2]  # new xmin
                flipped[:, 2] = new_w - boxes[:, 0]  # new xmax
                boxes = flipped

        # ── 5. To tensor + ImageNet normalisation ─────────────────────────
        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
        }
        return img, target


# ── Collate ────────────────────────────────────────────────────────────────
def custom_collate_fn(batch):
    """Return lists of images and targets (variable spatial sizes).

    Images have different H,W after dynamic resize; the train loop pads them
    to the batch maximum with pad_batch_images() in train.py.
    """
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ── Convenience builders ───────────────────────────────────────────────────
def build_trainval_dataset(root_dir: str, augment: bool = True) -> ConcatDataset:
    """VOC2007 trainval + VOC2012 trainval → 16,551 images (PDF Sec. 3).

    Args:
        root_dir: Path to VOCdevkit/.
        augment : Horizontal-flip augmentation (True for training).
    """
    voc07 = VOCDetectionDataset(
        root_dir, year="2007", image_set="trainval", augment=augment
    )
    voc12 = VOCDetectionDataset(
        root_dir, year="2012", image_set="trainval", augment=augment
    )
    combined = ConcatDataset([voc07, voc12])
    print(
        f"[Dataset] VOC2007 trainval: {len(voc07):,}  |  "
        f"VOC2012 trainval: {len(voc12):,}  |  "
        f"Combined: {len(combined):,}  (expected ~16,551)"
    )
    return combined


def build_test_dataset(root_dir: str) -> VOCDetectionDataset:
    """VOC2007 test set → 4,952 images (PDF Sec. 3). augment=False always."""
    ds = VOCDetectionDataset(
        root_dir, year="2007", image_set="test", augment=False
    )
    print(f"[Dataset] VOC2007 test: {len(ds):,}  (expected 4,952)")
    return ds
