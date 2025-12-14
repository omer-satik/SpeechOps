# scripts/inference.py
import torch
import librosa
import numpy as np
import soundfile as sf
import argparse

from src.model import UNet

# --- Configuration ---
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256

def denoise_audio(model_path, input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Load noisy audio
    noisy_wav, sr = librosa.load(input_path, sr=SAMPLE_RATE)
    print("Input audio loaded.")

    # Create spectrogram and phase
    noisy_spec = librosa.stft(noisy_wav, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag, noisy_phase = librosa.magphase(noisy_spec)
    
    # Convert data to model's expected format: [Batch, Channel, Frequency, Time]
    noisy_tensor = torch.from_numpy(noisy_mag).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        denoised_mag_tensor = model(noisy_tensor)

    # Convert data back to CPU and NumPy
    denoised_mag = denoised_mag_tensor.cpu().squeeze(0).squeeze(0).numpy()
    
    # Reconstruct the audio waveform
    denoised_wav = librosa.istft(denoised_mag * noisy_phase, hop_length=HOP_LENGTH, length=len(noisy_wav))
    print("Denoising complete.")

    # Save the cleaned audio
    sf.write(output_path, denoised_wav, SAMPLE_RATE)
    print(f"Cleaned audio saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Denoise an audio file using the trained U-Net model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pth file).")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the noisy input audio file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the denoised output audio file.")
    args = parser.parse_args()

    denoise_audio(args.model_path, args.input_file, args.output_file)

if __name__ == "__main__":
    main()
