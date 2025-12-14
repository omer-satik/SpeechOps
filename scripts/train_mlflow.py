# scripts/train_mlflow.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import mlflow
import mlflow.pytorch
import numpy as np
from datetime import datetime

from src.data_loader import SpectrogramDataset, pad_collate_fn
from src.model import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="SpeechOps Training with MLflow")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer type")
    parser.add_argument("--unet_depth", type=int, default=32, help="UNet base channel depth")
    parser.add_argument("--experiment_name", type=str, default="SpeechOps-UNet", help="MLflow experiment name")
    return parser.parse_args()


def calculate_snr(estimate: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio (SNR)."""
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
            
            # Compute SNR
            for i in range(outputs.shape[0]):
                snr = calculate_snr(outputs[i], clean_batch[i])
                if snr != float('inf'):
                    total_snr += snr
                    count += 1
    
    avg_loss = total_loss / len(val_loader)
    avg_snr = total_snr / count if count > 0 else 0.0
    model.train()
    return avg_loss, avg_snr


def main():
    args = parse_args()
    
    # Directories
    TRAIN_DIR = "data/processed/train"
    VAL_DIR = "data/processed/val"
    MODEL_SAVE_PATH = "models/"
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not torch.backends.mps.is_available() else torch.device("mps")
    print(f"Using device: {device}")
    
    # MLflow setup
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log hyperparameters
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("optimizer", args.optimizer)
        mlflow.log_param("unet_depth", args.unet_depth)
        mlflow.log_param("device", str(device))
        
        # Data loaders
        train_dataset = SpectrogramDataset(data_dir=TRAIN_DIR)
        val_dataset = SpectrogramDataset(data_dir=VAL_DIR)
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, collate_fn=pad_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, collate_fn=pad_collate_fn
        )
        
        mlflow.log_param("train_samples", len(train_dataset))
        mlflow.log_param("val_samples", len(val_dataset))
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        
        # Model
        model = UNet(in_channels=1, out_channels=1).to(device)
        criterion = nn.MSELoss()
        
        # Optimizer
        if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        
        # Training loop
        best_val_loss = float('inf')
        print("Starting training with MLflow tracking...")
        
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
                
                if (i + 1) % 20 == 0:
                    print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
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
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                print(f"  >> New best model saved! (val_loss: {val_loss:.4f})")
        
        print("\nFinished Training.")
        
        # Log final model to MLflow
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        # Save model as MLflow artifact
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="SpeechOps-UNet"
        )
        print(f"Model logged to MLflow Model Registry as 'SpeechOps-UNet'")
        
        # Save final model
        final_model_path = os.path.join(MODEL_SAVE_PATH, f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
