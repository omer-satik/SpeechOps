# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import librosa
from tqdm import tqdm

# Metrik kütüphaneleri
from pystoi import stoi

# Kendi modüllerimiz
from src.data_loader import SpectrogramDataset, pad_collate_fn
from src.model import UNet

# --- Yapılandırma ---
TEST_DATA_DIR = "data/processed/test"
MODEL_PATH = "models/baseline_unet1.pth"
BATCH_SIZE = 8 # Değerlendirme için batch size
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256

# Kendi SNR fonksiyonumuz
def signal_to_noise_ratio(estimate, target):
    """
    Sinyal-Gürültü Oranını (SNR) hesaplar.
    Formül: 10 * log10( ||target||^2 / ||target - estimate||^2 )
    """
    # Tensörlerin float olduğundan emin ol
    estimate = estimate.float()
    target = target.float()

    # Gürültü (hata) sinyalini hesapla
    noise = target - estimate

    # Sinyalin ve gürültünün gücünü (karelerinin toplamı) hesapla
    power_target = torch.sum(target ** 2)
    power_noise = torch.sum(noise ** 2)

    # Sıfıra bölünmeyi önlemek için küçük bir epsilon ekle
    if power_noise == 0:
        return float('inf') # Gürültü yoksa SNR sonsuzdur
    
    snr = 10 * torch.log10(power_target / power_noise)
    return snr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Modeli yükle
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # ÇOK ÖNEMLİ: Modeli değerlendirme moduna al
    print(f"Model loaded from {MODEL_PATH}")

    # Test veri setini yükle
    test_dataset = SpectrogramDataset(data_dir=TEST_DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn)
    print(f"Found {len(test_dataset)} samples for evaluation.")

    total_snr = 0
    total_stoi = 0
    count = 0

    print("Starting evaluation...")
    with torch.no_grad():  # Gradyan hesaplamasını kapatarak belleği ve hızı optimize et
        for noisy_batch, clean_batch in tqdm(test_loader):
            noisy_batch = noisy_batch.to(device)
            
            # Modelden temizlenmiş spektrogramı al
            denoised_spec_batch = model(noisy_batch).cpu().squeeze(1).numpy()
            
            # Orijinal temiz ve gürültülü spektrogramları da CPU'ya al
            clean_spec_batch = clean_batch.cpu().squeeze(1).numpy()
            noisy_spec_batch = noisy_batch.cpu().squeeze(1).numpy()

            for i in range(denoised_spec_batch.shape[0]):
                denoised_spec = denoised_spec_batch[i]
                clean_spec = clean_spec_batch[i]
                noisy_spec = noisy_spec_batch[i]
                
                # ÖNEMLİ: Ses dalgasını yeniden oluşturmak için FAZ bilgisine ihtiyacımız var.
                # Gürültülü sesin fazını "ödünç alıyoruz".
                # (Bu adım için önce gürültülü sesi yeniden oluşturmamız gerekir)
                noisy_wav = librosa.istft(noisy_spec, hop_length=HOP_LENGTH)
                _, noisy_phase = librosa.magphase(librosa.stft(noisy_wav, n_fft=N_FFT, hop_length=HOP_LENGTH))
                
                # Temizlenmiş ses dalgasını yeniden oluştur
                denoised_wav = librosa.istft(denoised_spec * noisy_phase, hop_length=HOP_LENGTH, length=len(noisy_wav))
                clean_wav = librosa.istft(clean_spec * noisy_phase, hop_length=HOP_LENGTH, length=len(noisy_wav))

                # Metrikleri hesapla
                # STOI için seslerin aynı uzunlukta olması gerekir
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