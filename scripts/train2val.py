import os
import shutil

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"

# Validation'a ayrılacak konuşmacılar
VAL_SPEAKERS = ["p256", "p278", "p243"]

def main():
    # Validation klasörünü oluştur
    os.makedirs(VAL_DIR, exist_ok=True)

    moved_count = 0

    # Train klasöründeki tüm dosyaları tara
    for filename in os.listdir(TRAIN_DIR):
        if not filename.endswith(".npz"):
            continue

        # Dosya ismi (p256_XXX.npz gibi)
        speaker_id = filename.split("_")[0]

        if speaker_id in VAL_SPEAKERS:
            src = os.path.join(TRAIN_DIR, filename)
            dst = os.path.join(VAL_DIR, filename)

            shutil.move(src, dst)
            moved_count += 1

    print(f"Validation setine taşınan dosya sayısı: {moved_count}")

if __name__ == "__main__":
    main()
