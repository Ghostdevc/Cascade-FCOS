import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import os
from PIL import Image

# Pascal VOC 20 classes
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

class VOCDetectionDataset(Dataset):
    def __init__(self, root_dir, year='2012', image_set='trainval', min_size=800, max_size=1333):
        """
        Args:
            root_dir: Path to the VOCdevkit folder.
            year: '2007' or '2012'
            image_set: 'train', 'val', or 'trainval'
            min_size: Target size for the short edge (e.g., 800 for FCOS P6/P7 resolution)
            max_size: Maximum size for the long edge to prevent OOM
        """
        self.root_dir = os.path.join(root_dir, f'VOC{year}')
        self.image_set = image_set
        self.min_size = min_size
        self.max_size = max_size
        
        # Load image IDs
        imgsets_file = os.path.join(self.root_dir, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(imgsets_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
            
        self.img_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.ann_dir = os.path.join(self.root_dir, 'Annotations')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # 2. Parse XML Annotations
        ann_path = os.path.join(self.ann_dir, f'{img_id}.xml')
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # FCOS usually ignores difficult objects in standard training, but we parse them
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
                
            cls_name = obj.find('name').text
            if cls_name not in CLASS_TO_IDX:
                continue
                
            bndbox = obj.find('bndbox')
            # Pascal VOC XML is 1-indexed, so we subtract 1 for 0-indexed math
            xmin = float(bndbox.find('xmin').text) - 1
            ymin = float(bndbox.find('ymin').text) - 1
            xmax = float(bndbox.find('xmax').text) - 1
            ymax = float(bndbox.find('ymax').text) - 1
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[cls_name])
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # 3. Dynamic Resize Logic (800px short edge)
        scale = self.min_size / min(w, h)
        if max(w, h) * scale > self.max_size:
            scale = self.max_size / max(w, h)
            
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Apply transforms
        img = TF.resize(img, (new_h, new_w))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Scale bounding boxes
        if boxes.numel() > 0:
            boxes = boxes * scale
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        
        return img, target

# Collate function is required because images and boxes have different sizes in a batch
def custom_collate_fn(batch):
    """
    Since images are dynamically resized, they might have different dimensions in a batch.
    Standard PyTorch practice for detection is to either pad images to the max size in the batch,
    or just return lists. We return lists here, which is standard for custom detection loops.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# --- Usage Example ---
# dataset = VOCDetectionDataset(root_dir='data/VOCdevkit', year='2012', image_set='trainval')
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)