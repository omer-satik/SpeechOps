# scripts/train_mlflow.py
"""
Training script with MLflow integration and configurable model architecture.
Supports different model sizes and hyperparameter experiments.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import mlflow
import mlflow.pytorch
from datetime import datetime

from src.data_loader import SpectrogramDataset, pad_collate_fn
from src.model import UNet, MODEL_CONFIGS, create_model


def parse_args():
    parser = argparse.ArgumentParser(description="SpeechOps Training with MLflow")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", 
                        choices=["adam", "adamw", "sgd"], help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["none", "step", "cosine"], help="Learning rate scheduler")
    
    # Model configuration
    parser.add_argument("--model_config", type=str, default="small",
                        choices=list(MODEL_CONFIGS.keys()), 
                        help="Predefined model config: small, medium, large, xlarge")
    parser.add_argument("--base_channels", type=int, default=None,
                        help="Override base channels (32, 64, 128)")
    parser.add_argument("--depth", type=int, default=None, choices=[2, 3],
                        help="Override model depth (2 or 3)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout rate (0.0-0.5)")
    
    # MLflow
    parser.add_argument("--experiment_name", type=str, default="SpeechOps-UNet",
                        help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name (default: auto-generated)")
    
    # Data
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    return parser.parse_args()


def calculate_snr(estimate: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in dB."""
    estimate = estimate.float()
    target = target.float()
    noise = target - estimate
    power_target = torch.sum(target ** 2)
    power_noise = torch.sum(noise ** 2)
    if power_noise == 0:
        return float('inf')
    snr = 10 * torch.log10(power_target / power_noise)
    return snr.item()


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple:
    """Validation step - returns avg_loss and avg_snr."""
    model.eval()
    total_loss = 0.0
    total_snr = 0.0
    count = 0
    
    with torch.no_grad():
        for noisy_batch, clean_batch in val_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            
            outputs = model(noisy_batch)
            loss = criterion(outputs, clean_batch)
            total_loss += loss.item()
            
            for i in range(outputs.shape[0]):
                snr = calculate_snr(outputs[i], clean_batch[i])
                if snr != float('inf'):
                    total_snr += snr
                    count += 1
    
    avg_loss = total_loss / len(val_loader)
    avg_snr = total_snr / count if count > 0 else 0.0
    model.train()
    return avg_loss, avg_snr


def get_optimizer(model, args):
    """Create optimizer based on args."""
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


def get_scheduler(optimizer, args, steps_per_epoch: int):
    """Create learning rate scheduler based on args."""
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3, gamma=0.1)
    elif args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return None


def main():
    args = parse_args()
    
    # Directories
    TRAIN_DIR = "data/processed/train"
    VAL_DIR = "data/processed/val"
    MODEL_SAVE_PATH = "models/"
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Build model config from args
    model_kwargs = {}
    if args.base_channels is not None:
        model_kwargs["base_channels"] = args.base_channels
    if args.depth is not None:
        model_kwargs["depth"] = args.depth
    if args.dropout is not None:
        model_kwargs["dropout"] = args.dropout
    
    # Create model
    model = create_model(args.model_config, **model_kwargs)
    model = model.to(device)
    
    # Get actual model config for logging
    actual_config = MODEL_CONFIGS[args.model_config].copy()
    actual_config.update(model_kwargs)
    
    num_params = model.count_parameters()
    print(f"Model config: {args.model_config} | Params: {num_params:,}")
    print(f"  base_channels={actual_config['base_channels']}, depth={actual_config['depth']}, dropout={actual_config['dropout']}")
    
    # MLflow setup
    mlflow.set_experiment(args.experiment_name)
    
    run_name = args.run_name or f"{args.model_config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log all hyperparameters
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("optimizer", args.optimizer)
        mlflow.log_param("weight_decay", args.weight_decay)
        mlflow.log_param("scheduler", args.scheduler)
        mlflow.log_param("model_config", args.model_config)
        mlflow.log_param("base_channels", actual_config["base_channels"])
        mlflow.log_param("depth", actual_config["depth"])
        mlflow.log_param("dropout", actual_config["dropout"])
        mlflow.log_param("num_parameters", num_params)
        mlflow.log_param("device", str(device))
        
        # Data loaders
        train_dataset = SpectrogramDataset(data_dir=TRAIN_DIR)
        val_dataset = SpectrogramDataset(data_dir=VAL_DIR)
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=pad_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=pad_collate_fn
        )
        
        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("val_samples", len(val_dataset))
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        
        # Loss, optimizer, scheduler
        criterion = nn.MSELoss()
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args, len(train_loader))
        
        # Training loop
        best_val_loss = float('inf')
        print("\nStarting training with MLflow tracking...")
        
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            
            for i, (noisy_batch, clean_batch) in enumerate(train_loader):
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(noisy_batch)
                loss = criterion(outputs, clean_batch)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (i + 1) % 50 == 0:
                    print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            # Update scheduler
            if scheduler:
                scheduler.step()
                mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
            
            # Epoch metrics
            train_loss = running_loss / len(train_loader)
            val_loss, val_snr = validate(model, val_loader, criterion, device)
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_snr", val_snr, step=epoch)
            
            print(f"---- Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SNR: {val_snr:.2f} dB ----")
            
            # Best model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": actual_config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_snr": val_snr
                }, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                print(f"  >> New best model saved! (val_loss: {val_loss:.4f})")
        
        print("\nFinished Training.")
        
        # Log final metrics
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        # Save model to MLflow
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"SpeechOps-{args.model_config}"
        )
        print(f"Model logged to MLflow Model Registry as 'SpeechOps-{args.model_config}'")
        
        # Save final model with config
        final_model_path = os.path.join(
            MODEL_SAVE_PATH, 
            f"{args.model_config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": actual_config,
            "args": vars(args)
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
