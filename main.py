import sys
try:
    import lzma
except ImportError:
    from backports import lzma
    sys.modules['lzma'] = lzma

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our custom Pure PyTorch modules
from datasets.pascal_voc import VOCDetectionDataset, custom_collate_fn
from models.cascade_fcos import CascadeFCOS
from models.target_generator import DynamicFCOSTargetGenerator
from models.loss import FCOSLoss
from train import train_one_epoch, save_checkpoint

def main(args):
    # 1. Environment Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Cascade FCOS Engine on device: {device}")
    
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"run_{args.run_name}.log")
    
    with open(log_file, "w") as f:
        f.write(f"--- Starting Cascade FCOS Training: {args.run_name} ---\n")

    # 2. Data Loading (Pascal VOC with Dynamic 800px Resizing)
    print("Loading Dataset...")
    train_dataset = VOCDetectionDataset(root_dir=args.data_dir, year='2012', image_set='trainval')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 3. Model & Optimizers
    print("Building Cascade FCOS Architecture...")
    model = CascadeFCOS(num_classes=20).to(device)
    
    # AdamW provides better weight decay handling than standard Adam
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning Rate Scheduler (Standard Object Detection practice: reduce LR at epochs 8 and 11)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)

    target_gen = DynamicFCOSTargetGenerator()
    loss_fn = FCOSLoss(alpha=0.25, gamma=2.0)

    # 4. Master Training Loop
    print("Commencing Training...")
    for epoch in range(1, args.epochs + 1):
        
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, target_gen, loss_fn, device, epoch, log_file
        )
        
        # Step the scheduler
        lr_scheduler.step()
        
        # Checkpoint and Log
        epoch_msg = f"=== Epoch {epoch} Completed | Average Loss: {avg_loss:.4f} ===\n"
        print(epoch_msg.strip())
        with open(log_file, "a") as f:
            f.write(epoch_msg)
            
        save_checkpoint(epoch, model, optimizer, avg_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascade FCOS Pure PyTorch Implementation")
    parser.add_argument("--data_dir", type=str, default="data/VOCdevkit", help="Path to Pascal VOC")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (4 recommended for 12GB VRAM)")
    parser.add_argument("--epochs", type=int, default=12, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--run_name", type=str, default="refinement_k1", help="Name of the experiment log")
    
    args = parser.parse_args()
    main(args)