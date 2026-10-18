"""
unet_model.py - Physics-Induced U-Net (PI-UNet / TopoUNet) for Real-Time Topology Optimization
Maps 6-channel physical boundary condition and load tensors directly to 1-channel optimal density fields.

Input: (B, 6, 60, 120) [or (B, 5, 60, 120) auto-upgraded on-the-fly]
  Channel 0: Domain solid mask (1.0 = solid design domain)
  Channel 1: bc_x (Ux = 0 horizontal support constraint mask)
  Channel 2: bc_y (Uy = 0 vertical support constraint mask)
  Channel 3: Fx (Horizontal load field)
  Channel 4: Fy (Vertical load field)
  Channel 5: Vf (Target volume fraction in [0.22, 0.55])

Output: (B, 1, 60, 120)
  Predicted optimal structural density field in [0, 1] with exact volume enforcement.

Key Architectural Modules:
  1. PhysicsFeatureStem: On-the-fly multi-scale Gaussian/Green's potentials (sigma=9, 21)
     and normalized Cartesian coordinate channels (6 -> 12 dense physics channels).
  2. ResFiLMBlock: Residual DoubleConv + Squeeze-and-Excitation (SE) + FiLM conditioning on [Vf, 1-Vf, ||F||].
  3. LoadPathSelfAttention: 4-head Multi-Head Self-Attention at the 8x16 bottleneck.
  4. ConicalSIMPFilter: Fixed Sigmund rmin=1.5 conical filter layer.
  5. DifferentiableVolumeProjection: 6-step Newton-Raphson OC bisection layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableGaussianFilter2D(nn.Module):
    """
    Separable 2D Gaussian filter implemented via sequential 1D convolutions.
    Non-trainable, deterministic, and supports replicate padding for zero boundary artifacts.
    """
    def __init__(self, sigma):
        super().__init__()
        self.sigma = float(sigma)
        radius = max(1, int(math.ceil(3.0 * self.sigma)))
        kernel_size = 2 * radius + 1
        self.radius = radius

        # Compute 1D Gaussian kernel
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (coords / self.sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Reshape for conv2d: (out_channels, in_channels/groups, kH, kW)
        k_horiz = kernel_1d.view(1, 1, 1, kernel_size)
        k_vert = kernel_1d.view(1, 1, kernel_size, 1)

        self.register_buffer('k_horiz', k_horiz)
        self.register_buffer('k_vert', k_vert)

    def forward(self, x):
        # x shape: (B, 1, H, W)
        p = self.radius
        # Horizontal pass
        x_pad_h = F.pad(x, (p, p, 0, 0), mode='replicate')
        x_h = F.conv2d(x_pad_h, self.k_horiz)
        # Vertical pass
        x_pad_v = F.pad(x_h, (0, 0, p, p), mode='replicate')
        out = F.conv2d(x_pad_v, self.k_vert)
        return out


class PhysicsFeatureStem(nn.Module):
    """
    Physics-Induced Feature Stem:
    Upgrades 5-channel or 6-channel input into 12 dense physics channels on-the-fly:
      - 6 original channels: [domain, bc_x, bc_y, Fx, Fy, Vf]
      - 4 Green's potential fields:
          * phi_bc_s9:  Gaussian potential on bc_mag with sigma = 9.0
          * phi_bc_s21: Gaussian potential on bc_mag with sigma = 21.0
          * phi_fx_s9:  Gaussian potential on Fx with sigma = 9.0
          * phi_fy_s9:  Gaussian potential on Fy with sigma = 9.0
      - 2 Normalized spatial coordinate channels: x_hat in [-1, 1], y_hat in [-1, 1]
    Total output channels: 6 + 4 + 2 = 12 channels.

    Also extracts scalar conditioning vector: [Vf, 1 - Vf, ||F||] of shape (B, 3).
    """
    def __init__(self):
        super().__init__()
        self.filter_s9 = SeparableGaussianFilter2D(sigma=9.0)
        self.filter_s21 = SeparableGaussianFilter2D(sigma=21.0)

    def forward(self, x):
        # Auto-upgrade 5-channel input to 6-channel
        if x.shape[1] == 5:
            # [domain, bc, Fx, Fy, Vf] -> [domain, bc, bc, Fx, Fy, Vf]
            x = torch.cat([x[:, 0:1], x[:, 1:2], x[:, 1:2], x[:, 2:5]], dim=1)

        b, c, h, w = x.shape
        domain = x[:, 0:1]
        bc_x = x[:, 1:2]
        bc_y = x[:, 2:3]
        fx = x[:, 3:4]
        fy = x[:, 4:5]
        vf = x[:, 5:6]

        # 1. Support magnitude mask
        bc_mag = torch.clamp(bc_x + bc_y, 0.0, 1.0)

        # 2. Multi-scale Green's potential fields
        phi_bc_s9 = self.filter_s9(bc_mag)
        phi_bc_s21 = self.filter_s21(bc_mag)
        phi_fx_s9 = self.filter_s9(fx)
        phi_fy_s9 = self.filter_s9(fy)

        # 3. Normalized spatial coordinates in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        coord_x = grid_x.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
        coord_y = grid_y.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)

        # Concatenate 12 dense channels
        dense_physics = torch.cat([
            x,
            phi_bc_s9,
            phi_bc_s21,
            phi_fx_s9,
            phi_fy_s9,
            coord_x,
            coord_y
        ], dim=1)

        # 4. Scalar conditioning vector: [Vf, 1 - Vf, ||F||] (shape B, 3)
        vf_mean = vf.mean(dim=(-2, -1)) # (B, 1)
        vf_comp = 1.0 - vf_mean         # (B, 1)
        f_norm = torch.sqrt(fx**2 + fy**2).sum(dim=(-2, -1)) # (B, 1)
        cond = torch.cat([vf_mean, vf_comp, f_norm], dim=1)  # (B, 3)

        return dense_physics, cond, vf_mean.unsqueeze(-1).unsqueeze(-1)


class ResFiLMBlock(nn.Module):
    """
    Residual DoubleConv + Squeeze-and-Excitation (SE) + FiLM Conditioning Block.
    
    Sequence:
      Conv3x3 -> BatchNorm -> SiLU -> Conv3x3 -> BatchNorm
      FiLM Modulation: h = h * (1.0 + gamma) + beta
      Squeeze-and-Excitation (SE) channel recalibration
      Residual 1x1 shortcut addition + SiLU
    """
    def __init__(self, in_channels, out_channels, cond_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # FiLM linear projection: predicts (gamma, beta) from condition embedding
        self.film = nn.Linear(cond_dim, 2 * out_channels)

        # Squeeze-and-Excitation (SE) attention
        se_reduced = max(out_channels // 8, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, se_reduced),
            nn.SiLU(inplace=True),
            nn.Linear(se_reduced, out_channels),
            nn.Sigmoid()
        )

        # 1x1 residual shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x, cond_emb):
        residual = self.shortcut(x)

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.bn2(h)

        # FiLM modulation: gamma and beta
        film_params = self.film(cond_emb).unsqueeze(-1).unsqueeze(-1) # (B, 2*C, 1, 1)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        h = h * (1.0 + gamma) + beta

        # Squeeze-and-Excitation attention
        se_weights = self.se(h).unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        h = h * se_weights

        # Residual connection
        out = self.act2(h + residual)
        return out


class DownBlock(nn.Module):
    """
    Downsampling block: MaxPool2d(2x2) followed by ResFiLMBlock.
    """
    def __init__(self, in_channels, out_channels, cond_dim=64):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ResFiLMBlock(in_channels, out_channels, cond_dim=cond_dim)

    def forward(self, x, cond_emb):
        x = self.pool(x)
        return self.conv(x, cond_emb)


class UpBlock(nn.Module):
    """
    Upsampling block: Bilinear upsample / ConvTranspose2d followed by skip connection
    concatenation and ResFiLMBlock.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, cond_dim=64):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResFiLMBlock(in_channels, out_channels, cond_dim=cond_dim)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResFiLMBlock(in_channels, out_channels, cond_dim=cond_dim)

    def forward(self, x1, x2, cond_emb):
        x1 = self.up(x1)
        # Pad if spatial dimensions differ slightly
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, cond_emb)


class LoadPathSelfAttention(nn.Module):
    """
    4-Head Multi-Head Self-Attention at the Bottleneck (8x16 spatial tokens = 128 elements).
    Allows distant load points (e.g. at x = 120) to directly attend to support points (at x = 0).
    """
    def __init__(self, embed_dim, num_heads=4, num_tokens=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_tokens = num_tokens

        # Positional encoding: initialized with 2D sinusoidal pattern
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self._init_pos_embed()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def _init_pos_embed(self):
        # 2D grid coordinates for 8 x 16 tokens
        h, w = 8, 16
        dim = self.embed_dim
        pe = torch.zeros(h, w, dim)
        d_half = dim // 2
        omega = torch.exp(torch.arange(0, d_half, 2, dtype=torch.float32) * -(math.log(10000.0) / d_half))

        pos_y = torch.arange(h, dtype=torch.float32).unsqueeze(1)
        pos_x = torch.arange(w, dtype=torch.float32).unsqueeze(1)

        sin_y = torch.sin(pos_y * omega)
        cos_y = torch.cos(pos_y * omega)
        sin_x = torch.sin(pos_x * omega)
        cos_x = torch.cos(pos_x * omega)

        pe_y = torch.cat([sin_y, cos_y], dim=1) # (h, d_half)
        pe_x = torch.cat([sin_x, cos_x], dim=1) # (w, d_half)

        pe[:, :, :d_half] = pe_y.unsqueeze(1).expand(h, w, d_half)
        pe[:, :, d_half:] = pe_x.unsqueeze(0).expand(h, w, d_half)

        pe_flat = pe.view(1, h * w, dim)
        with torch.no_grad():
            self.pos_embed.copy_(pe_flat)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2) # (B, H*W, C)

        # Add positional embedding (handle matching or interpolated token counts)
        if tokens.shape[1] == self.num_tokens:
            tokens = tokens + self.pos_embed
        else:
            # Interpolate pos_embed if dimensions ever differ
            pe = self.pos_embed.transpose(1, 2).view(1, c, 8, 16)
            pe_interp = F.interpolate(pe, size=(h, w), mode='bilinear', align_corners=False)
            pe_interp = pe_interp.flatten(2).transpose(1, 2)
            tokens = tokens + pe_interp

        # Self-Attention block
        norm_tokens = self.norm1(tokens)
        attn_out, _ = self.attn(norm_tokens, norm_tokens, norm_tokens)
        tokens = tokens + attn_out

        # FFN block
        tokens = tokens + self.ffn(self.norm2(tokens))

        # Reshape back to spatial map
        out = tokens.transpose(1, 2).view(b, c, h, w)
        return out


class ConicalSIMPFilter(nn.Module):
    """
    Fixed, non-trainable 3x3 Sigmund conical filter kernel for rmin = 1.5.
    H_ij = max(0, 1.5 - sqrt(i^2 + j^2)) for i, j in {-1, 0, 1}.
    Applied with replicate padding to prevent boundary erosion.
    """
    def __init__(self, rmin=1.5):
        super().__init__()
        self.rmin = rmin

        kernel = torch.zeros(3, 3, dtype=torch.float32)
        for i in range(-1, 2):
            for j in range(-1, 2):
                dist = math.sqrt(i**2 + j**2)
                kernel[i + 1, j + 1] = max(0.0, rmin - dist)
        kernel = kernel / kernel.sum()
        self.register_buffer('weight', kernel.view(1, 1, 3, 3))

    def forward(self, x):
        # x shape: (B, 1, H, W)
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(x_pad, self.weight)


class DifferentiableVolumeProjection(nn.Module):
    """
    6-step unrolled damped Newton-Raphson OC bisection layer.
    Solves for scalar shift mu in R^{B x 1 x 1 x 1} such that:
      mean(sigmoid(z_tilde + mu)) = Vf_target
    """
    def __init__(self, num_steps=6, damping=2.5):
        super().__init__()
        self.num_steps = num_steps
        self.damping = damping

    def forward(self, z_tilde, vf_target, domain_mask=None):
        if vf_target.ndim == 2:
            vf_target = vf_target.unsqueeze(-1).unsqueeze(-1)
        vf_target = torch.clamp(vf_target, 0.05, 0.95)

        # Initial mu estimate
        mu = torch.logit(vf_target) - z_tilde.mean(dim=(-2, -1), keepdim=True)

        for _ in range(self.num_steps):
            p = torch.sigmoid(z_tilde + mu)
            if domain_mask is not None:
                mask_sum = domain_mask.sum(dim=(-2, -1), keepdim=True) + 1e-6
                res = (p * domain_mask).sum(dim=(-2, -1), keepdim=True) / mask_sum - vf_target
                deriv = ((p * (1.0 - p)) * domain_mask).sum(dim=(-2, -1), keepdim=True) / mask_sum + 1e-6
            else:
                res = p.mean(dim=(-2, -1), keepdim=True) - vf_target
                deriv = (p * (1.0 - p)).mean(dim=(-2, -1), keepdim=True) + 1e-6
            step = torch.clamp(res / deriv, -self.damping, self.damping)
            mu = mu - step

        out = torch.sigmoid(z_tilde + mu)
        return out


class TopoUNet(nn.Module):
    """
    Physics-Induced U-Net (PI-UNet / TopoUNet) for Real-Time Topology Optimization.
    
    Default parameters:
      in_channels = 6 (accepts 6-channel or 5-channel tensors)
      out_channels = 1
      base_filters = 64
      bilinear = True
      enforce_volume = True
      use_simp_filter = True
    """
    def __init__(self, in_channels=6, out_channels=1, base_filters=64,
                 bilinear=True, enforce_volume=True, use_simp_filter=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.enforce_volume = enforce_volume
        self.use_simp_filter = use_simp_filter

        # 1. Physics Feature Stem: Maps (6 or 5) channels to 12 dense channels
        self.stem = PhysicsFeatureStem()

        # Conditioning MLP: [Vf, 1 - Vf, ||F||] (dim 3) -> 64
        cond_dim = 64
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim)
        )

        # 2. ResFiLM Encoder
        self.inc = ResFiLMBlock(12, base_filters, cond_dim=cond_dim)
        self.down1 = DownBlock(base_filters, base_filters * 2, cond_dim=cond_dim)
        self.down2 = DownBlock(base_filters * 2, base_filters * 4, cond_dim=cond_dim)
        self.down3 = DownBlock(base_filters * 4, base_filters * 8, cond_dim=cond_dim)

        # 3. Load Path Self-Attention at the 8x16 Bottleneck
        self.bottleneck_attn = LoadPathSelfAttention(base_filters * 8, num_heads=4, num_tokens=128)

        # 4. ResFiLM Decoder
        factor = 2 if bilinear else 1
        self.up1 = UpBlock(base_filters * 8 + base_filters * 4, (base_filters * 4) // factor, bilinear, cond_dim=cond_dim)
        self.up2 = UpBlock((base_filters * 4) // factor + base_filters * 2, (base_filters * 2) // factor, bilinear, cond_dim=cond_dim)
        self.up3 = UpBlock((base_filters * 2) // factor + base_filters, base_filters, bilinear, cond_dim=cond_dim)

        # Prediction head
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

        # 5. Conical SIMP filter and Volume Projection
        self.simp_filter = ConicalSIMPFilter(rmin=1.5)
        self.vol_proj = DifferentiableVolumeProjection(num_steps=6)

        # Cache for loss regularization
        self.last_raw_probs = None

    def forward(self, x):
        orig_h, orig_w = x.shape[-2], x.shape[-1]

        # Extract 12 dense physics channels and scalar conditioning
        dense_physics, cond, vf_target = self.stem(x)
        cond_emb = self.cond_mlp(cond)

        # Pad to multiple of 16 for clean 3-level pooling (e.g. 60x120 -> 64x128)
        pad_h = (16 - orig_h % 16) % 16
        pad_w = (16 - orig_w % 16) % 16
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(dense_physics, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        else:
            x_padded = dense_physics

        # Encoder forward pass
        x1 = self.inc(x_padded, cond_emb)
        x2 = self.down1(x1, cond_emb)
        x3 = self.down2(x2, cond_emb)
        x4 = self.down3(x3, cond_emb)

        # Bottleneck self-attention
        x4 = self.bottleneck_attn(x4)

        # Decoder forward pass with skip connections
        x_dec = self.up1(x4, x3, cond_emb)
        x_dec = self.up2(x_dec, x2, cond_emb)
        x_dec = self.up3(x_dec, x1, cond_emb)

        logits = self.outc(x_dec)

        # Crop back to original spatial dimensions (orig_h, orig_w)
        if pad_h > 0 or pad_w > 0:
            logits = logits[:, :, pad_top:pad_top + orig_h, pad_left:pad_left + orig_w]

        # Conical SIMP filter
        if self.use_simp_filter:
            z_tilde = self.simp_filter(logits)
        else:
            z_tilde = logits

        # Store pre-projection probabilities for loss regularization
        self.last_raw_probs = torch.sigmoid(z_tilde)

        # Exact volume projection
        if self.enforce_volume:
            domain_mask = x[:, 0:1] if x.shape[1] >= 1 else None
            out = self.vol_proj(z_tilde, vf_target, domain_mask=domain_mask)
        else:
            out = self.last_raw_probs

        return out


# Alias for Physics-Induced U-Net
PIUNet = TopoUNet


def count_parameters(model):
    """Returns total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
