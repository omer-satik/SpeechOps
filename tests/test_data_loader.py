# tests/test_data_loader.py
import torch
import pytest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import SpectrogramDataset, pad_collate_fn


class TestSpectrogramDataset:
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with sample .npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample spectrograms
            for i in range(3):
                noisy = np.random.randn(257, 100 + i * 10).astype(np.float32)
                clean = np.random.randn(257, 100 + i * 10).astype(np.float32)
                np.savez(os.path.join(tmpdir, f"sample_{i}.npz"), noisy=noisy, clean=clean)
            yield tmpdir

    def test_dataset_length(self, temp_data_dir):
        """Test dataset returns correct number of samples."""
        dataset = SpectrogramDataset(temp_data_dir)
        assert len(dataset) == 3

    def test_dataset_output_type(self, temp_data_dir):
        """Test dataset returns torch tensors."""
        dataset = SpectrogramDataset(temp_data_dir)
        noisy, clean = dataset[0]
        assert isinstance(noisy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)

    def test_dataset_output_shape(self, temp_data_dir):
        """Test dataset returns tensors with channel dimension."""
        dataset = SpectrogramDataset(temp_data_dir)
        noisy, clean = dataset[0]
        assert noisy.dim() == 3  # [C, H, W]
        assert noisy.shape[0] == 1  # Single channel


class TestPadCollateFn:
    def test_padding_to_max_length(self):
        """Test collate function pads to max length."""
        batch = [
            (torch.randn(1, 257, 50), torch.randn(1, 257, 50)),
            (torch.randn(1, 257, 100), torch.randn(1, 257, 100)),
            (torch.randn(1, 257, 75), torch.randn(1, 257, 75)),
        ]
        noisy_batch, clean_batch = pad_collate_fn(batch)
        
        assert noisy_batch.shape == (3, 1, 257, 100)
        assert clean_batch.shape == (3, 1, 257, 100)

    def test_same_length_no_padding(self):
        """Test collate function works with same-length inputs."""
        batch = [
            (torch.randn(1, 257, 100), torch.randn(1, 257, 100)),
            (torch.randn(1, 257, 100), torch.randn(1, 257, 100)),
        ]
        noisy_batch, clean_batch = pad_collate_fn(batch)
        
        assert noisy_batch.shape == (2, 1, 257, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

