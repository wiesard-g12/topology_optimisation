"""
unet_model.py - 4-Level Encoder-Decoder U-Net for Topology Optimization
Maps 5-channel physical input tensors directly to 1-channel optimal density fields.

Input: (B, 5, 60, 120)
  Channel 0: Domain solid mask
  Channel 1: Boundary support locations (pins/rollers)
  Channel 2: Horizontal load field Fx
  Channel 3: Vertical load field Fy
  Channel 4: Target volume fraction Vf

Output: (B, 1, 60, 120)
  Predicted optimal structural density field in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard U-Net Double Convolution block:
    [Conv2d -> BatchNorm -> LeakyReLU(0.2)] x 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block: MaxPool2d(2x2) followed by DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block: Bilinear / Transpose Conv followed by skip concatenation and DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: from decoder (needs upsampling)
        # x2: skip connection from encoder
        x1 = self.up(x1)
        
        # In case of minor spatial size differences due to odd dimensions:
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TopoUNet(nn.Module):
    """
    Physics-informed U-Net for 2D Structural Topology Optimization.
    
    Default architecture:
      Level 1: 5 -> 64
      Level 2: 64 -> 128
      Level 3: 128 -> 256
      Bottleneck: 256 -> 512
      Decoder 3: 512 + 256 -> 256
      Decoder 2: 256 + 128 -> 128
      Decoder 1: 128 + 64  -> 64
      Final Head: 64 -> 1 with Sigmoid activation
    
    Grid handling:
      Automatically pads (60, 120) to (64, 128) during forward pass
      so downsampling cleanly divides by 2 at all 4 levels, then crops
      back to exact (60, 120) for zero spatial distortion.
    """
    def __init__(self, in_channels=5, out_channels=1, base_filters=64, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_filters)                           # -> 64
        self.down1 = DownBlock(base_filters, base_filters * 2)                    # -> 128
        self.down2 = DownBlock(base_filters * 2, base_filters * 4)                # -> 256
        self.down3 = DownBlock(base_filters * 4, base_filters * 8)                # -> 512
        
        # Decoder
        factor = 2 if bilinear else 1
        self.up1 = UpBlock(base_filters * 8 + base_filters * 4, base_filters * 4 // factor, bilinear)
        self.up2 = UpBlock(base_filters * 4 // factor + base_filters * 2, base_filters * 2 // factor, bilinear)
        self.up3 = UpBlock(base_filters * 2 // factor + base_filters, base_filters, bilinear)
        
        # Output prediction head
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        
        # Pad to multiple of 16 for clean 3-level pooling (e.g. 60x120 -> 64x128)
        pad_h = (16 - orig_h % 16) % 16
        pad_w = (16 - orig_w % 16) % 16
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        else:
            x_padded = x
            
        # Encoder forward pass
        x1 = self.inc(x_padded)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder forward pass with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        logits = self.outc(x)
        out = self.sigmoid(logits)
        
        # Crop back to original spatial dimensions (H, W)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, pad_top:pad_top + orig_h, pad_left:pad_left + orig_w]
            
        return out


def count_parameters(model):
    """Returns total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    model = TopoUNet()
    tot, trn = count_parameters(model)
    print(f"TopoUNet initialized successfully!")
    print(f"Total parameters: {tot:,} | Trainable: {trn:,}")
    
    # Test tensor forward pass with (B, 5, 60, 120)
    dummy_input = torch.randn(2, 5, 60, 120)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 1, 60, 120), "Shape mismatch!"
    print(f"Output range: [{dummy_output.min().item():.4f}, {dummy_output.max().item():.4f}]")
    print("Verification PASSED!")
