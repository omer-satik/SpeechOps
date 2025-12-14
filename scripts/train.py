# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import our custom modules
from src.data_loader import SpectrogramDataset, pad_collate_fn
from src.model import UNet

# --- Hyperparameters and Configuration ---
DATA_DIR = "data/processed/train"
MODEL_SAVE_PATH = "models/"
NUM_EPOCHS = 5  # Small value for initial testing
BATCH_SIZE = 4
LEARNING_RATE = 0.001

def main():
    # Create the directory for saving models
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Set device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the dataset and DataLoader
    dataset = SpectrogramDataset(data_dir=DATA_DIR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=pad_collate_fn)
    print(f"Found {len(dataset)} samples for training.")

    # Define model, loss function and optimizer
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error, good starting point for spectrograms
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (noisy_batch, clean_batch) in enumerate(train_loader):
            # Send data to selected device
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(noisy_batch)
            
            # Compute loss
            loss = criterion(outputs, clean_batch)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Log every 10 batches
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f"---- Epoch {epoch+1} finished. Average Loss: {running_loss / len(train_loader):.4f} ----")
    
    print("Finished Training.")
    
    # Save the trained model
    final_model_path = os.path.join(MODEL_SAVE_PATH, "baseline_unet.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
