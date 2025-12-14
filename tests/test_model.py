# tests/test_model.py
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import UNet, DoubleConv


class TestDoubleConv:
    def test_output_shape(self):
        """Test DoubleConv maintains spatial dimensions."""
        conv = DoubleConv(1, 32)
        x = torch.randn(1, 1, 64, 128)
        out = conv(x)
        assert out.shape == (1, 32, 64, 128)

    def test_channel_change(self):
        """Test DoubleConv correctly changes channels."""
        conv = DoubleConv(32, 64)
        x = torch.randn(1, 32, 64, 128)
        out = conv(x)
        assert out.shape[1] == 64


class TestUNet:
    def test_output_shape_matches_input(self):
        """Test UNet output has same spatial dimensions as input."""
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 256, 128)
        out = model(x)
        assert out.shape == x.shape

    def test_batch_processing(self):
        """Test UNet handles multiple samples in batch."""
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(4, 1, 128, 64)
        out = model(x)
        assert out.shape[0] == 4

    def test_variable_input_sizes(self):
        """Test UNet handles different input sizes."""
        model = UNet(in_channels=1, out_channels=1)
        for size in [(64, 32), (128, 64), (256, 128)]:
            x = torch.randn(1, 1, size[0], size[1])
            out = model(x)
            assert out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

