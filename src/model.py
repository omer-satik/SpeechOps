# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Configurable U-Net for speech enhancement.
    
    Args:
        in_channels: Number of input channels (default: 1 for magnitude spectrogram)
        out_channels: Number of output channels (default: 1)
        base_channels: Base number of channels (controls model capacity)
                       - 32: Small model (~0.5M params) - baseline1
                       - 64: Medium model (~2M params) - baseline2
                       - 128: Large model (~8M params)
        depth: Number of down/up sampling levels (2 or 3)
        dropout: Dropout rate (0.0 = no dropout)
    """
    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 1, 
        base_channels: int = 32,
        depth: int = 2,
        dropout: float = 0.0
    ):
        super(UNet, self).__init__()
        self.depth = depth
        self.base_channels = base_channels
        
        # Channel progression: base -> base*2 -> base*4 -> ...
        c = base_channels  # shorthand
        
        # Encoder
        self.inc = DoubleConv(in_channels, c, dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c*2, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*2, c*4, dropout))
        
        if depth == 3:
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*4, c*8, dropout))
            self.up0 = nn.ConvTranspose2d(c*8, c*4, kernel_size=2, stride=2)
            self.conv0 = DoubleConv(c*8, c*4, dropout)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(c*4, c*2, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(c*4, c*2, dropout)
        self.up2 = nn.ConvTranspose2d(c*2, c, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(c*2, c, dropout)
        
        # Output
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)
    
    def _pad_to_match(self, x, target):
        """Pad x to match target's spatial dimensions."""
        diffY = target.size()[2] - x.size()[2]
        diffX = target.size()[3] - x.size()[3]
        return F.pad(x, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        if self.depth == 3:
            x4 = self.down3(x3)
            x = self.up0(x4)
            x = self._pad_to_match(x, x3)
            x = torch.cat([x, x3], dim=1)
            x = self.conv0(x)
        else:
            x = x3
        
        # Decoder
        x = self.up1(x)
        x = self._pad_to_match(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self._pad_to_match(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        
        return self.outc(x)
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Pre-defined model configurations
MODEL_CONFIGS = {
    "small": {"base_channels": 32, "depth": 2, "dropout": 0.0},    # ~0.5M params
    "medium": {"base_channels": 64, "depth": 2, "dropout": 0.0},   # ~2M params  
    "large": {"base_channels": 64, "depth": 3, "dropout": 0.1},    # ~8M params
    "xlarge": {"base_channels": 128, "depth": 3, "dropout": 0.1},  # ~32M params
}


def create_model(config_name: str = "small", **kwargs) -> UNet:
    """
    Factory function to create UNet with predefined configs.
    
    Args:
        config_name: One of 'small', 'medium', 'large', 'xlarge'
        **kwargs: Override any config parameter
    
    Returns:
        UNet model instance
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[config_name].copy()
    config.update(kwargs)
    
    return UNet(**config)
