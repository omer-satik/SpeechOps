# scripts/evaluate.py
"""
Evaluate trained models on test set.
Supports loading models with different configurations.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import librosa
from tqdm import tqdm
import argparse
import mlflow

from pystoi import stoi
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr

from src.data_loader import SpectrogramDataset, pad_collate_fn
from src.model import UNet, MODEL_CONFIGS, create_model

# --- Configuration ---
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SpeechOps model")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, default="data/processed/test",
                        help="Path to test data directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Model config (only needed for old .pth files without config)
    parser.add_argument("--model_config", type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model config (auto-detected from checkpoint if available)")
    parser.add_argument("--base_channels", type=int, default=None,
                        help="Base channels (for old checkpoints)")
    
    # MLflow logging
    parser.add_argument("--log_mlflow", action="store_true",
                        help="Log results to MLflow")
    parser.add_argument("--experiment_name", type=str, default="SpeechOps-Evaluation",
                        help="MLflow experiment name")
    
    return parser.parse_args()

def load_model(model_path: str, args, device: torch.device) -> tuple:
    """
    Load model from checkpoint, auto-detecting config if available.
    Returns: (model, config_dict)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint contains model config (new format)
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        config = checkpoint["model_config"]
        model = UNet(
            base_channels=config.get("base_channels", 32),
            depth=config.get("depth", 2),
            dropout=config.get("dropout", 0.0)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model with config: {config}")
        return model.to(device), config
    
    # Old format: just state_dict
    # Use provided args or defaults
    if args.model_config:
        model = create_model(args.model_config)
        config = MODEL_CONFIGS[args.model_config].copy()
    elif args.base_channels:
        model = UNet(base_channels=args.base_channels)
        config = {"base_channels": args.base_channels, "depth": 2, "dropout": 0.0}
    else:
        # Default to small
        model = UNet(base_channels=32)
        config = {"base_channels": 32, "depth": 2, "dropout": 0.0}
    
    # Load weights (handle both formats)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded model with inferred config: {config}")
    return model.to(device), config


def evaluate(model, test_loader, device):
    """Run evaluation and return metrics."""
    model.eval()
    
    total_snr = 0.0
    total_stoi = 0.0
    count = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for noisy_batch, clean_batch in tqdm(test_loader):
            noisy_batch = noisy_batch.to(device)
            
            denoised_spec_batch = model(noisy_batch).cpu().squeeze(1).numpy()
            clean_spec_batch = clean_batch.cpu().squeeze(1).numpy()
            noisy_spec_batch = noisy_batch.cpu().squeeze(1).numpy()

            for i in range(denoised_spec_batch.shape[0]):
                denoised_spec = denoised_spec_batch[i]
                clean_spec = clean_spec_batch[i]
                noisy_spec = noisy_spec_batch[i]
                
                # Reconstruct waveform using noisy phase
                noisy_wav = librosa.istft(noisy_spec, hop_length=HOP_LENGTH)
                _, noisy_phase = librosa.magphase(
                    librosa.stft(noisy_wav, n_fft=N_FFT, hop_length=HOP_LENGTH)
                )
                
                denoised_wav = librosa.istft(
                    denoised_spec * noisy_phase, 
                    hop_length=HOP_LENGTH, 
                    length=len(noisy_wav)
                )
                clean_wav = librosa.istft(
                    clean_spec * noisy_phase, 
                    hop_length=HOP_LENGTH, 
                    length=len(noisy_wav)
                )

                # Ensure same length for STOI
                min_len = min(len(clean_wav), len(denoised_wav))
                clean_wav = clean_wav[:min_len]
                denoised_wav = denoised_wav[:min_len]

                total_stoi += stoi(clean_wav, denoised_wav, SAMPLE_RATE, extended=False)
                total_snr += si_snr(torch.from_numpy(denoised_wav), torch.from_numpy(clean_wav)).item()
                count += 1
    
    return {
        "avg_snr": total_snr / count,
        "avg_stoi": total_stoi / count,
        "num_samples": count
    }


def main():
    args = parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, model_config = load_model(args.model_path, args, device)
    model.eval()
    print(f"Model loaded from {args.model_path}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Load test data
    test_dataset = SpectrogramDataset(data_dir=args.test_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=pad_collate_fn
    )
    print(f"Found {len(test_dataset)} samples for evaluation.")
    
    # Evaluate
    metrics = evaluate(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Config: base_channels={model_config.get('base_channels')}, depth={model_config.get('depth')}")
    print(f"Test samples: {metrics['num_samples']}")
    print("-"*50)
    print(f"Average SNR:  {metrics['avg_snr']:.4f} dB")
    print(f"Average STOI: {metrics['avg_stoi']:.4f}")
    print("="*50)
    
    # Log to MLflow
    if args.log_mlflow:
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name=f"eval_{args.model_path.split('/')[-1]}"):
            mlflow.log_param("model_path", args.model_path)
            mlflow.log_param("base_channels", model_config.get("base_channels"))
            mlflow.log_param("depth", model_config.get("depth"))
            mlflow.log_param("num_samples", metrics["num_samples"])
            mlflow.log_metric("avg_snr", metrics["avg_snr"])
            mlflow.log_metric("avg_stoi", metrics["avg_stoi"])
            print("\nResults logged to MLflow.")


if __name__ == "__main__":
    main()
