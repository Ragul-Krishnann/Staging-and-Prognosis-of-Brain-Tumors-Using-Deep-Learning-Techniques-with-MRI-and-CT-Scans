import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast 
from tqdm import tqdm
import os

# Import custom modules
from dataset import BrainTumorDataset
from models.yolov7_dual import YOLOv7_DualFusion

# --- 1. CONFIGURATION ---
CONFIG = {
    # Absolute path to handle Windows environment
    "base_path": r"C:\Users\ragul\Desktop\BrainTumor_Project\data",
    "batch_size": 16,
    "learning_rate": 0.0001,
    "epochs": 100,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "./weights/brain_tumor_final.pth"
}

def train():
    # Ensure weights directory exists
    os.makedirs("./weights", exist_ok=True)
    
    # --- 2. DATA PREPARATION ---
    # Implements Layer 1: Data Preparation
    train_ds = BrainTumorDataset(CONFIG['base_path'], split='train')
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True # Speeds up data transfer to GPU
    )

    # --- 3. MODEL INITIALIZATION ---
    # Dual-stream YOLOv7 with CBAM and BiFPN
    model = YOLOv7_DualFusion(num_classes=4).to(CONFIG['device'])
    
    # --- 4. LOSS & OPTIMIZER ---
    # Multi-task loss for staging (Classification) and prognosis
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # GPU Scaler for Mixed Precision training
    scaler = GradScaler() 

    # Identify GPU name for confirmation
    if CONFIG['device'].type == 'cuda':
        device_name = torch.cuda.get_device_name(0)
        print(f"--- Training started on GPU: {device_name} ---")
    else:
        print("--- WARNING: CUDA not found. Training on CPU will be extremely slow. ---")

    # --- 5. TRAINING LOOP ---
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for mri, ct, labels in pbar:
            # Move data to GPU memory
            mri, ct, labels = mri.to(CONFIG['device']), ct.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()

            # Accelerated Forward Pass (AMP)
            with autocast():
                # Forward Pass: F_Fused = alpha * F_MRI + (1 - alpha) * F_CT
                cls_out, prog_out = model(mri, ct)
                loss = criterion_cls(cls_out, labels)
            
            # Accelerated Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics for logging
            running_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Alpha": f"{model.alpha.item():.3f}" # Monitor learnable fusion parameter
            })

        # --- 6. CHECKPOINTING ---
        # Save weights every 10 epochs or when finished
        if (epoch + 1) % 10 == 0 or (epoch + 1) == CONFIG['epochs']:
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"\n[INFO] Checkpoint saved at epoch {epoch+1}")

    print("\n--- Training Finalized ---")
    print(f"Final Model saved at: {CONFIG['save_path']}")

if __name__ == "__main__":
    train()