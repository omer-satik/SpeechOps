# scripts/train2val.py
import os
import shutil

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"

# Speakers to be moved to validation set
VAL_SPEAKERS = ["p256", "p278", "p243"]

def main():
    # Create validation directory
    os.makedirs(VAL_DIR, exist_ok=True)

    moved_count = 0

    # Scan all files in train directory
    for filename in os.listdir(TRAIN_DIR):
        if not filename.endswith(".npz"):
            continue

        # Filename format: (p256_XXX.npz)
        speaker_id = filename.split("_")[0]

        if speaker_id in VAL_SPEAKERS:
            src = os.path.join(TRAIN_DIR, filename)
            dst = os.path.join(VAL_DIR, filename)

            shutil.move(src, dst)
            moved_count += 1

    print(f"Number of files moved to validation set: {moved_count}")

if __name__ == "__main__":
    main()
