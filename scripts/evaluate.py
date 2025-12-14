# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import librosa
from tqdm import tqdm

# Metric libraries
from pystoi import stoi

# Our custom modules
from src.data_loader import SpectrogramDataset, pad_collate_fn
from src.model import UNet

# --- Configuration ---
TEST_DATA_DIR = "data/processed/test"
MODEL_PATH = "models/baseline_unet1.pth"
BATCH_SIZE = 8  # Batch size for evaluation
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256

# Custom SNR function
def signal_to_noise_ratio(estimate, target):
    """
    Compute Signal-to-Noise Ratio (SNR).
    Formula: 10 * log10( ||target||^2 / ||target - estimate||^2 )
    """
    # Ensure tensors are float
    estimate = estimate.float()
    target = target.float()

    # Compute noise (error) signal
    noise = target - estimate

    # Compute power (sum of squares) of signal and noise
    power_target = torch.sum(target ** 2)
    power_noise = torch.sum(noise ** 2)

    # Add small epsilon to prevent division by zero
    if power_noise == 0:
        return float('inf')  # SNR is infinite if there's no noise
    
    snr = 10 * torch.log10(power_target / power_noise)
    return snr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # IMPORTANT: Set model to evaluation mode
    print(f"Model loaded from {MODEL_PATH}")

    # Load test dataset
    test_dataset = SpectrogramDataset(data_dir=TEST_DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn)
    print(f"Found {len(test_dataset)} samples for evaluation.")

    total_snr = 0
    total_stoi = 0
    count = 0

    print("Starting evaluation...")
    with torch.no_grad():  # Disable gradient computation to optimize memory and speed
        for noisy_batch, clean_batch in tqdm(test_loader):
            noisy_batch = noisy_batch.to(device)
            
            # Get denoised spectrogram from model
            denoised_spec_batch = model(noisy_batch).cpu().squeeze(1).numpy()
            
            # Move original clean and noisy spectrograms to CPU
            clean_spec_batch = clean_batch.cpu().squeeze(1).numpy()
            noisy_spec_batch = noisy_batch.cpu().squeeze(1).numpy()

            for i in range(denoised_spec_batch.shape[0]):
                denoised_spec = denoised_spec_batch[i]
                clean_spec = clean_spec_batch[i]
                noisy_spec = noisy_spec_batch[i]
                
                # IMPORTANT: We need PHASE information to reconstruct audio waveform.
                # We "borrow" the phase from the noisy audio.
                noisy_wav = librosa.istft(noisy_spec, hop_length=HOP_LENGTH)
                _, noisy_phase = librosa.magphase(librosa.stft(noisy_wav, n_fft=N_FFT, hop_length=HOP_LENGTH))
                
                # Reconstruct the denoised audio waveform
                denoised_wav = librosa.istft(denoised_spec * noisy_phase, hop_length=HOP_LENGTH, length=len(noisy_wav))
                clean_wav = librosa.istft(clean_spec * noisy_phase, hop_length=HOP_LENGTH, length=len(noisy_wav))

                # Compute metrics
                # STOI requires audio signals to have the same length
                min_len = min(len(clean_wav), len(denoised_wav))
                clean_wav = clean_wav[:min_len]
                denoised_wav = denoised_wav[:min_len]

                total_stoi += stoi(clean_wav, denoised_wav, SAMPLE_RATE, extended=False)
                total_snr += signal_to_noise_ratio(torch.from_numpy(denoised_wav), torch.from_numpy(clean_wav)).item()
                count += 1
    
    avg_snr = total_snr / count
    avg_stoi = total_stoi / count
    
    print("\n--- Evaluation Finished ---")
    print(f"Average Signal-to-Noise Ratio (SNR): {avg_snr:.4f} dB")
    print(f"Average Short-Time Objective Intelligibility (STOI): {avg_stoi:.4f}")

if __name__ == "__main__":
    main()
