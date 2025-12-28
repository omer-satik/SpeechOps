# SpeechOps

Speech Enhancement with MLOps - A U-Net based audio denoising system with complete MLOps pipeline.

## Project Structure

```
speechops/
├── api/                    # FastAPI service for deployment (Phase 3)
│   └── __init__.py
├── configs/                # Configuration files
│   └── config.yaml         # Main configuration
├── data/                   # Data directory (DVC tracked)
│   ├── raw/                # Original datasets
│   ├── processed/          # Preprocessed spectrograms
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── *.dvc               # DVC tracking files
├── logs/                   # Log files
├── mlruns/                 # MLflow experiment tracking
├── models/                 # Trained model weights
├── notebooks/              # Jupyter notebooks for EDA
├── scripts/                # Training and utility scripts
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Basic training script
│   ├── train_mlflow.py     # Training with MLflow tracking
│   ├── evaluate.py         # Model evaluation
│   ├── inference.py        # Single file inference
│   └── train2val.py        # Train/val split utility
├── src/                    # Source code package
│   ├── __init__.py
│   ├── data_loader.py      # PyTorch Dataset and DataLoader
│   └── model.py            # U-Net model architecture
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_model.py
│   └── test_data_loader.py
├── .dvc/                   # DVC configuration
├── .gitignore
├── pyproject.toml          # Project configuration
├── README.md
└── requirements.txt        # Python dependencies
```

## Setup

### 1. Create Virtual Environment

```bash
conda create -n speechops python=3.11 -y
conda activate speechops
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Data (DVC)

```bash
dvc pull
```

## Usage

### Preprocessing

```bash
python scripts/preprocess.py
```

### Training with MLflow

```bash
# Start MLflow UI (in separate terminal)
mlflow ui --host 127.0.0.1 --port 5001

# Run training
PYTHONPATH=. python scripts/train_mlflow.py --epochs 10 --lr 0.001 --batch_size 8
```

### Evaluation

```bash
PYTHONPATH=. python scripts/evaluate.py
```

### Inference

```bash
PYTHONPATH=. python scripts/inference.py \
    --model-path models/best_model.pth \
    --input-file noisy_audio.wav \
    --output-file clean_audio.wav
```

### Run Tests

```bash
pytest tests/ -v
```

## Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t speechops-service .

# Run the container
docker run -p 8080:8080 -v ./models:/app/models:ro speechops-service
```

### Using Docker Compose

```bash
# Start API service
docker-compose up -d

# Start with MLflow UI (optional)
docker-compose --profile full up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/predict` | Denoise audio file (returns WAV) |
| POST | `/predict/json` | Denoise and return metadata |

### Test with cURL

```bash
# Health check
curl http://127.0.0.1:8080/health

# Denoise audio file
curl -X POST "http://127.0.0.1:8080/predict" \
  -H "accept: audio/wav" \
  -F "file=@noisy_audio.wav" \
  --output clean_audio.wav
```

## MLOps Pipeline

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project setup, Git, DVC | ✅ |
| 1 | Data pipeline, baseline model | ✅ |
| 2 | Experiment tracking (MLflow) | ✅ |
| 3 | Deployment (FastAPI + Docker) | 🔜 |
| 4 | CI/CD and Monitoring | 🔜 |

## Configuration

Edit `configs/config.yaml` for hyperparameters:

```yaml
training:
  epochs: 10
  batch_size: 8
  learning_rate: 0.001
  optimizer: "adam"
```

## Model Architecture

U-Net encoder-decoder with skip connections for spectrogram-based speech enhancement.

- **Input**: Noisy magnitude spectrogram [1, 257, T]
- **Output**: Enhanced magnitude spectrogram [1, 257, T]
- **Loss**: MSE Loss
- **Metrics**: SI-SNR, STOI

## License

MIT

