# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Kendi modüllerimizi import edelim
from src.data_loader import SpectrogramDataset, pad_collate_fn
from src.model import UNet

# --- Hiperparametreler ve Yapılandırma ---
DATA_DIR = "data/processed/train"
MODEL_SAVE_PATH = "models/"
NUM_EPOCHS = 5  # Başlangıç için küçük bir değer
BATCH_SIZE = 4
LEARNING_RATE = 0.001

def main():
    # Modelin kaydedileceği klasörü oluştur
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Cihazı ayarla (GPU varsa kullan, yoksa CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Veri setini ve DataLoader'ı oluştur
    dataset = SpectrogramDataset(data_dir=DATA_DIR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=pad_collate_fn)
    print(f"Found {len(dataset)} samples for training.")

    # Modeli, kayıp fonksiyonunu ve optimizer'ı tanımla
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error, spektrogramlar için iyi bir başlangıç
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (noisy_batch, clean_batch) in enumerate(train_loader):
            # Veriyi seçilen cihaza gönder
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            # Optimizer'ın gradyanlarını sıfırla
            optimizer.zero_grad()

            # İleri besleme (forward pass)
            outputs = model(noisy_batch)
            
            # Kaybı hesapla
            loss = criterion(outputs, clean_batch)

            # Geri yayılım (backward pass) ve ağırlıkları güncelle
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Her 10 batch'te bir log yazdır
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f"---- Epoch {epoch+1} finished. Average Loss: {running_loss / len(train_loader):.4f} ----")
    
    print("Finished Training.")
    
    # Eğitilmiş modeli kaydet
    final_model_path = os.path.join(MODEL_SAVE_PATH, "baseline_unet.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    main()