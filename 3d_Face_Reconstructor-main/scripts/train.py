import sys
import os
from pathlib import Path
import argparse
import multiprocessing as mp
import shutil
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import  GradScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.network import FaceReconstructionNet
from scripts.dataset import FaceDataset

def load_config(path: str):
    p = Path(path)
    if not p.exists():
        # Default config
        print(f"⚠️ Config file {path} not found, using default configuration")
        return {
            "epochs": 20,
            "batch_size": 4,  # Reduced to prevent OOM
            "lr": 1e-4,
            "num_workers": 2,
            "save_every": 1,
            "img_dir": "data/processed/images",
            "coeff_dir": "data/processed/coeffs",
            "ckpt_dir": "models/checkpoints",
            "tri_uvs_path": "data/tri_uvs.json",
            "uv_loss_weight": 1.0,
            "accum_steps": 4,
            "mixed_precision": True
        }
    
    # ✅ Load and safely parse YAML with type conversion
    with open(p, "r") as f:
        config_data = yaml.safe_load(f)
    
    # ✅ Ensure all numeric values are proper types
    if config_data is None:
        config_data = {}
    
    # Convert string numbers to proper types
    numeric_keys = ["epochs", "batch_size", "lr", "num_workers", "save_every", 
                   "uv_loss_weight", "accum_steps"]
    
    for key in numeric_keys:
        if key in config_data and isinstance(config_data[key], str):
            try:
                if '.' in config_data[key]:
                    config_data[key] = float(config_data[key])
                else:
                    config_data[key] = int(config_data[key])
            except ValueError:
                print(f"⚠️ Warning: Could not convert {key}={config_data[key]} to number, using default")
                # Use default if conversion fails
                default_config = {
                    "epochs": 20,
                    "batch_size": 4,
                    "lr": 1e-4,
                    "num_workers": 2,
                    "save_every": 1,
                    "uv_loss_weight": 1.0,
                    "accum_steps": 4
                }
                if key in default_config:
                    config_data[key] = default_config[key]
    
    # Ensure boolean values
    if "mixed_precision" in config_data and isinstance(config_data["mixed_precision"], str):
        config_data["mixed_precision"] = config_data["mixed_precision"].lower() in ["true", "1", "yes"]
    
    print(f"✅ Loaded configuration from {path}")
    return config_data

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def save_checkpoint(epoch, model, optimizer, scaler, loss, filepath):
    """Save complete training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Saved checkpoint: {filepath} (epoch {epoch})")

def load_checkpoint(filepath, model, optimizer, scaler, device):
    """Load complete training state"""
    if not os.path.exists(filepath):
        return None, 0
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"🔄 Loaded checkpoint: {filepath} (resuming from epoch {start_epoch})")
    return start_epoch, loss

def train(cfg):
    print(f"🔧 Configuration: {cfg}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 Training device:", device)

    dataset = FaceDataset(cfg["img_dir"], cfg["coeff_dir"], cfg["tri_uvs_path"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

    coeff_dim = 228
    num_triangles = dataset.tri_uvs.shape[0] if dataset.tri_uvs is not None else 116160
    print(f"[Info] Number of triangles for network: {num_triangles}")
    model = FaceReconstructionNet(coeff_dim=coeff_dim, num_triangles=num_triangles).to(device)

    # ✅ ADD DEBUG CODE RIGHT HERE:
    print("🔍 DEBUGGING DATA RANGES:")
    for i, (imgs, coeffs, uv) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  Images - min: {imgs.min():.6f}, max: {imgs.max():.6f}, mean: {imgs.mean():.6f}")
        print(f"  Coeffs - min: {coeffs.min():.6f}, max: {coeffs.max():.6f}, mean: {coeffs.mean():.6f}")
        print(f"  UVs - min: {uv.min():.6f}, max: {uv.max():.6f}, mean: {uv.mean():.6f}")
        
        if abs(coeffs.max()) > 5.0 or abs(coeffs.min()) > 5.0:
          print("❌ COEFFICIENTS STILL TOO LARGE AFTER SCALING!")
    
        if i >= 1:  # Only check first 2 batches
          break
    print("--- Debug complete ---")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = torch.nn.MSELoss()

    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg.get("epochs", 20)
    save_every = cfg.get("save_every", 1)
    uv_loss_weight = cfg.get("uv_loss_weight", 1.0)
    accum_steps = cfg.get("accum_steps", 1)

    # ✅ CHECKPOINT RESUME LOGIC
    start_epoch = 0
    latest_ckpt = ckpt_dir / "latest.pth"
    
    # Try to resume from latest checkpoint
    if latest_ckpt.exists():
        print(f"🔄 Found existing checkpoint: {latest_ckpt}")
        resume = input("Do you want to resume training from checkpoint? (y/n): ").lower().strip()
        if resume == 'y':
            start_epoch, _ = load_checkpoint(latest_ckpt, model, optimizer, scaler, device)
            start_epoch += 1  # Start from next epoch
            print(f"🔄 Resuming training from epoch {start_epoch + 1}")
        else:
            print("🚀 Starting fresh training")
            start_epoch = 0
    else:
        print("🚀 No checkpoint found, starting fresh training")

    # ✅ MAIN TRAINING LOOP WITH PROPER EPOCH COUNTING
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
    
        # ✅ PRINT WITH CORRECT EPOCH NUMBER
        print(f"🎯 Starting Epoch {epoch+1}/{epochs}")
        print_gpu_memory()
    
        for i, (imgs, coeffs, uv) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            coeffs = coeffs.to(device, non_blocking=True)
            uv = uv.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=cfg.get("mixed_precision", True)):
                pred_coeff, pred_uv = model(imgs)
                loss_coeff = criterion(pred_coeff, coeffs)
                loss_uv = F.mse_loss(pred_uv, uv)
                loss = loss_coeff + uv_loss_weight * loss_uv
                
                # ✅ CLIP LOSS TO PREVENT EXPLOSION
                loss = torch.clamp(loss, max=1000.0)  # Prevent huge losses
                loss = loss / accum_steps
                
                if torch.isnan(loss).any():
                    print(f"❌ NaN detected at batch {i}! Skipping.")
                    optimizer.zero_grad()
                    continue

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

        # ✅ PRINT CLEAN EPOCH RESULTS
        avg_loss = total_loss / max(1, len(loader))
        print(f"✅ Epoch {epoch+1}/{epochs} COMPLETED — avg_loss: {avg_loss:.6f}")
        print_gpu_memory()
    
        # ✅ SAVE COMPLETE CHECKPOINT (not just model weights)
        if ((epoch+1) % save_every == 0) or (epoch+1 == epochs):
            # Save epoch-specific checkpoint
            epoch_ckpt = ckpt_dir / f"epoch_{epoch+1}.pth"
            save_checkpoint(epoch+1, model, optimizer, scaler, avg_loss, epoch_ckpt)
            
            # Update latest checkpoint
            latest_ckpt = ckpt_dir / "latest.pth"
            save_checkpoint(epoch+1, model, optimizer, scaler, avg_loss, latest_ckpt)
            
    print("✅ Training finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    ns = parser.parse_args()
    cfg = load_config(ns.config)

    mp.set_start_method("spawn", force=True)
    train(cfg)

if __name__ == "__main__":
    main()