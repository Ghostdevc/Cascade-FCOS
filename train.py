import os
import torch
import torch.nn.utils as utils
from models.target_generator import DynamicFCOSTargetGenerator
from models.loss import FCOSLoss
import torch.nn.functional as F

def pad_batch_images(images):
    """Pads a list of images to the max Height and Width in the batch."""
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])
    padded_imgs = []
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded_imgs.append(F.pad(img, (0, pad_w, 0, pad_h)))
    return torch.stack(padded_imgs)

def train_one_epoch(model, dataloader, optimizer, target_generator, loss_fn, device, epoch, log_file):
    model.train()
    total_epoch_loss = 0.0
    
    # The "Bonus Point" Cascade Logic: Dynamically shrinking the center sampling radius
    stage_radiuses = {"stage_1": 2.5, "stage_2": 1.5, "stage_3": 1.0}
    
    for batch_idx, (images_list, targets_list) in enumerate(dataloader):
        images = pad_batch_images(images_list).to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass (Extract FPN -> Stage 1 Head -> FCM -> Stage 2 Head -> FCM -> Stage 3 Head)
        cascade_preds = model(images)
        features = cascade_preds["features_s1"]
        
        # Base spatial mapping
        locations_per_level = target_generator.compute_locations(features)
        batch_loss = 0.0
        
        # Accumulate loss across all stages (Joint End-to-End Training)
        for stage_name, radius in stage_radiuses.items():
            cls_preds, reg_preds, cent_preds = cascade_preds[stage_name]
            
            flat_cls = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 20) for p in cls_preds], dim=0)
            flat_reg = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 4) for p in reg_preds], dim=0)
            flat_cent = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 1) for p in cent_preds], dim=0)
            
            batch_labels, batch_reg_targets = [], []
            for target in targets_list:
                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)
                
                labels, reg_targets = target_generator.generate_targets_for_image(
                    locations_per_level, gt_boxes, gt_labels, center_radius=radius
                )
                batch_labels.append(labels)
                batch_reg_targets.append(reg_targets)
                
            flat_labels = torch.cat(batch_labels, dim=0)
            flat_reg_targets = torch.cat(batch_reg_targets, dim=0)
            
            # Compute Tripartite Loss
            l_cls, l_reg, l_cent = loss_fn(flat_cls, flat_reg, flat_cent, flat_labels, flat_reg_targets)
            stage_loss = l_cls + l_reg + l_cent
            batch_loss += stage_loss
            
        # Backpropagation
        batch_loss.backward()
        
        # Security Measure: Prevent Exploding Gradients
        utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        
        total_epoch_loss += batch_loss.item()
        
        if batch_idx % 20 == 0:
            log_msg = f"Epoch: {epoch} | Batch {batch_idx}/{len(dataloader)} | Total Cascade Loss: {batch_loss.item():.4f}\n"
            print(log_msg.strip())
            with open(log_file, "a") as f:
                f.write(log_msg)

    avg_loss = total_epoch_loss / len(dataloader)
    return avg_loss

def save_checkpoint(epoch, model, optimizer, loss, save_dir="logs/checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"cascade_fcos_epoch_{epoch}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved securely to {checkpoint_path}")

# --- Entry Point ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CascadeFCOS().to(device)
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# target_gen = DynamicFCOSTargetGenerator()
# loss_fn = FCOSLoss()
# dataset = VOCDetectionDataset(...)
# dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)
# 
# for epoch in range(12): # Standard 12-epoch VOC schedule
#     train_one_epoch(model, dataloader, optimizer, target_gen, loss_fn, device)