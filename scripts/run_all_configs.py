#!/usr/bin/env python
"""
Run training for all model configurations with the same hyperparameters.
Logs all runs to MLflow for easy comparison.
"""
import subprocess
import sys
from datetime import datetime

# Model configurations to train
MODEL_CONFIGS = ["small", "medium", "large", "xlarge"]

# Common hyperparameters
COMMON_ARGS = {
    "epochs": 50,
    "lr": 0.0005,
    "batch_size": 16,
    "optimizer": "adamw",
    "weight_decay": 0.0001,
    "scheduler": "cosine",
    "dropout": 0.1,
    "experiment_name": "SpeechOps-AllConfigs",
}


def run_training(model_config: str) -> bool:
    """Run training for a single model configuration."""
    print(f"\n{'='*60}")
    print(f"Starting training for: {model_config.upper()}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "-m", "scripts.train_mlflow",
        f"--model_config={model_config}",
        f"--run_name={model_config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    ]
    
    # Add common arguments
    for key, value in COMMON_ARGS.items():
        cmd.append(f"--{key}={value}")
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {model_config} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_config} failed with error code {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {model_config} interrupted by user\n")
        return False


def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           SpeechOps - Training All Model Configs             ║
╠══════════════════════════════════════════════════════════════╣
║  Configs: {', '.join(MODEL_CONFIGS):<48} ║
║  Epochs: {COMMON_ARGS['epochs']:<49} ║
║  Learning Rate: {COMMON_ARGS['lr']:<42} ║
║  Optimizer: {COMMON_ARGS['optimizer']:<46} ║
║  Scheduler: {COMMON_ARGS['scheduler']:<46} ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    for config in MODEL_CONFIGS:
        success = run_training(config)
        results[config] = "✓ Success" if success else "✗ Failed"
        
        if not success:
            print(f"Training failed for {config}. Continue with next config? [Y/n]")
            try:
                response = input().strip().lower()
                if response == 'n':
                    break
            except EOFError:
                # Non-interactive mode, continue
                pass
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for config, status in results.items():
        print(f"  {config:<10} : {status}")
    print(f"{'='*60}")
    print(f"View results in MLflow UI: http://127.0.0.1:5001")
    print(f"Experiment: {COMMON_ARGS['experiment_name']}")


if __name__ == "__main__":
    main()
