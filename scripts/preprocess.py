# scripts/preprocess.py
import os
import librosa
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Define constants
PROCESSED_DATA_PATH = "data/processed"
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
HF_DATASET_NAME = "JacobLinCool/VoiceBank-DEMAND-16k"

def process_example(example, save_dir):
    """Process a single audio example from Hugging Face and save it."""
    try:
        # Get data using correct column names: 'clean' and 'noisy'
        clean_audio_data = example['clean']
        noisy_audio_data = example['noisy']
        file_id = example['id']

        clean_audio = clean_audio_data['array']
        noisy_audio = noisy_audio_data['array']
        sr = clean_audio_data['sampling_rate']

        # Resample if sample rate is different
        if sr != SAMPLE_RATE:
            clean_audio = librosa.resample(y=clean_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            noisy_audio = librosa.resample(y=noisy_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Create spectrograms
        clean_spec = np.abs(librosa.stft(clean_audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        noisy_spec = np.abs(librosa.stft(noisy_audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        
        # Determine filename and save path
        base_filename = file_id.replace('.wav', '')
        save_path = os.path.join(save_dir, f"{base_filename}.npz")
        
        # Save the file
        np.savez(save_path, noisy=noisy_spec, clean=clean_spec)
        
    except Exception as e:
        print(f"Error processing example {file_id}: {e}")

def main():
    print(f"Loading dataset '{HF_DATASET_NAME}' from Hugging Face Hub...")
    
    dataset = load_dataset(HF_DATASET_NAME, split="test")

    print("Starting data preprocessing...")
    # Create directories for processed data
    test_save_dir = os.path.join(PROCESSED_DATA_PATH, "test")
    os.makedirs(test_save_dir, exist_ok=True)
    
    # Process each example
    for example in tqdm(dataset, desc="Processing Test Set"):
        process_example(example, test_save_dir)

    print("Data preprocessing finished.")

if __name__ == "__main__":
    main()
