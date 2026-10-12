"""
generate_unet_diagram.py - Generates a comprehensive architectural diagram of TopoUNet
Illustrates the 5-channel physical input, 4-level encoder-decoder hierarchy,
skip connections, tensor dimensions at every stage, and the multi-objective loss.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch
import os

def create_unet_diagram(output_path="images/unet_architecture_details.png"):
    fig = plt.figure(figsize=(18, 10), dpi=200, facecolor='#0b0f19')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0b0f19')
    
    # Coordinates coordinate system: [0, 100] in X, [0, 60] in Y
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')
    
    # ----------------------------------------------------
    # Header & Title
    # ----------------------------------------------------
    ax.text(50, 57.5, "TopoUNet: Deep Learning Structural Topology Optimization", 
            ha='center', va='center', fontsize=20, fontweight='bold', color='#ffffff')
    ax.text(50, 54.8, "4-Level Encoder-Decoder Architecture with Symmetry Padding, Skip Connections & Physics Loss", 
            ha='center', va='center', fontsize=12, color='#94a3b8')
    
    # Helper to draw rounded boxes
    def draw_box(x, y, w, h, bg_color, border_color, title, subtitle="", text_color='#ffffff'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.8",
                             facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', 
                    fontsize=9.5, fontweight='bold', color=text_color, zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', 
                    fontsize=7.8, color='#cbd5e1', zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', 
                    fontsize=9.5, fontweight='bold', color=text_color, zorder=3)
        return (x + w/2, y + h/2)

    # Helper to draw arrows
    def draw_arrow(p1, p2, color='#38bdf8', style='->', lw=1.8, dashed=False):
        ls = '--' if dashed else '-'
        arrow = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=14,
                                color=color, linewidth=lw, linestyle=ls, zorder=4)
        ax.add_patch(arrow)

    # ----------------------------------------------------
    # 1. Input Tensor Block (Left)
    # ----------------------------------------------------
    draw_box(2, 38, 12, 13, '#1e293b', '#3b82f6', 
             "Input Physics State\n(5 Channels)", 
             "60 x 120 x 5\nSolid, BCs, Fx, Fy, Vf", '#60a5fa')
    
    # 5 Channels Breakdown sub-box
    ch_box = FancyBboxPatch((2, 21), 12, 14.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                            facecolor='#0f172a', edgecolor='#475569', linewidth=1.2, zorder=2)
    ax.add_patch(ch_box)
    ax.text(8, 33.5, "Input Channels Breakdown:", ha='center', va='center', 
            fontsize=8.5, fontweight='bold', color='#93c5fd')
    channels = [
        "Ch 0: Domain Solid Mask [0,1]",
        "Ch 1: Support Pins/Rollers",
        "Ch 2: Horizontal Force Fx",
        "Ch 3: Vertical Force Fy",
        "Ch 4: Target Volfrac Vf"
    ]
    for idx, ch in enumerate(channels):
        ax.text(3, 30.5 - idx*2.2, f"• {ch}", ha='left', va='center', 
                fontsize=7.2, color='#e2e8f0', zorder=3)

    # Arrow Input -> Padding
    draw_arrow((14.5, 44.5), (17.5, 44.5), color='#38bdf8')

    # Symmetric Reflection Padding
    draw_box(18, 41, 10, 7, '#1e293b', '#6366f1', 
             "Reflection Padding", "(60, 120) -> (64, 128)", '#a5b4fc')
    
    draw_arrow((28.5, 44.5), (31.5, 44.5), color='#38bdf8')

    # ----------------------------------------------------
    # 2. Encoder Path (Downwards Left)
    # ----------------------------------------------------
    # Level 1
    enc1 = draw_box(32, 41, 11, 7, '#1e1b4b', '#818cf8', 
                    "Encoder Level 1", "64 x 128 x 64\nDoubleConv (3x3)")
    
    # Down 1
    draw_arrow((37.5, 40.5), (37.5, 36.5), color='#f43f5e', lw=2)
    ax.text(41.5, 38.5, "MaxPool (2x2)", fontsize=7.5, color='#fca5a5', ha='center')

    # Level 2
    enc2 = draw_box(32, 29, 11, 7, '#1e1b4b', '#818cf8', 
                    "Encoder Level 2", "32 x 64 x 128\nDoubleConv (3x3)")
    
    # Down 2
    draw_arrow((37.5, 28.5), (37.5, 24.5), color='#f43f5e', lw=2)
    ax.text(41.5, 26.5, "MaxPool (2x2)", fontsize=7.5, color='#fca5a5', ha='center')

    # Level 3
    enc3 = draw_box(32, 17, 11, 7, '#1e1b4b', '#818cf8', 
                    "Encoder Level 3", "16 x 32 x 256\nDoubleConv (3x3)")
    
    # Down 3
    draw_arrow((37.5, 16.5), (37.5, 12.5), color='#f43f5e', lw=2)
    ax.text(41.5, 14.5, "MaxPool (2x2)", fontsize=7.5, color='#fca5a5', ha='center')

    # Level 4
    enc4 = draw_box(32, 5, 11, 7, '#1e1b4b', '#818cf8', 
                    "Encoder Level 4", "8 x 16 x 512\nDoubleConv (3x3)")
    
    # Down 4 to Bottleneck
    draw_arrow((43.5, 8.5), (46.5, 8.5), color='#f43f5e', lw=2)

    # ----------------------------------------------------
    # 3. Bottleneck (Center Bottom)
    # ----------------------------------------------------
    btn = draw_box(47, 5, 11, 7, '#3b0764', '#c084fc', 
                   "Bottleneck", "4 x 8 x 512\nDropout(0.1)", '#e9d5ff')

    draw_arrow((58.5, 8.5), (61.5, 8.5), color='#10b981', lw=2)

    # ----------------------------------------------------
    # 4. Decoder Path (Upwards Right)
    # ----------------------------------------------------
    # Level 4 Up
    dec4 = draw_box(62, 5, 11, 7, '#064e3b', '#34d399', 
                    "Decoder Level 4", "8 x 16 x 256\nConvTrans + Concat")
    
    # Up 1
    draw_arrow((67.5, 12.5), (67.5, 16.5), color='#10b981', lw=2)
    ax.text(71.5, 14.5, "ConvTranspose", fontsize=7.5, color='#6ee7b7', ha='center')

    # Level 3 Up
    dec3 = draw_box(62, 17, 11, 7, '#064e3b', '#34d399', 
                    "Decoder Level 3", "16 x 32 x 128\nConvTrans + Concat")
    
    # Up 2
    draw_arrow((67.5, 24.5), (67.5, 28.5), color='#10b981', lw=2)
    ax.text(71.5, 26.5, "ConvTranspose", fontsize=7.5, color='#6ee7b7', ha='center')

    # Level 2 Up
    dec2 = draw_box(62, 29, 11, 7, '#064e3b', '#34d399', 
                    "Decoder Level 2", "32 x 64 x 64\nConvTrans + Concat")
    
    # Up 3
    draw_arrow((67.5, 36.5), (67.5, 40.5), color='#10b981', lw=2)
    ax.text(71.5, 38.5, "ConvTranspose", fontsize=7.5, color='#6ee7b7', ha='center')

    # Level 1 Up
    dec1 = draw_box(62, 41, 11, 7, '#064e3b', '#34d399', 
                    "Decoder Level 1", "64 x 128 x 64\nFinal DoubleConv")

    # ----------------------------------------------------
    # 5. Skip Connections (Horizontal Dashed Lines)
    # ----------------------------------------------------
    # Skip 1
    draw_arrow((43.5, 44.5), (61.5, 44.5), color='#fbbf24', style='->', lw=1.6, dashed=True)
    ax.text(52.5, 45.8, "Skip Concatenation (64 ch)", fontsize=7.5, color='#fef08a', ha='center')

    # Skip 2
    draw_arrow((43.5, 32.5), (61.5, 32.5), color='#fbbf24', style='->', lw=1.6, dashed=True)
    ax.text(52.5, 33.8, "Skip Concatenation (128 ch)", fontsize=7.5, color='#fef08a', ha='center')

    # Skip 3
    draw_arrow((43.5, 20.5), (61.5, 20.5), color='#fbbf24', style='->', lw=1.6, dashed=True)
    ax.text(52.5, 21.8, "Skip Concatenation (256 ch)", fontsize=7.5, color='#fef08a', ha='center')

    # Skip 4
    draw_arrow((43.5, 8.5), (61.5, 8.5), color='#fbbf24', style='->', lw=1.6, dashed=True)

    # ----------------------------------------------------
    # 6. Output Head & Crop (Right)
    # ----------------------------------------------------
    draw_arrow((73.5, 44.5), (76.5, 44.5), color='#38bdf8')
    
    # 1x1 Conv + Sigmoid
    draw_box(77, 41, 9, 7, '#1e293b', '#06b6d4', 
             "1x1 Conv + Sigmoid", "(64, 128) x 1 Head", '#67e8f9')
    
    draw_arrow((86.5, 44.5), (88.5, 44.5), color='#38bdf8')

    # Center Crop to (60, 120)
    draw_box(89, 41, 9, 7, '#1e293b', '#10b981', 
             "Center Crop", "-> exact (60, 120)", '#6ee7b7')
    
    # Final Output Density Box
    out_box = draw_box(85, 25, 13, 11, '#1e293b', '#22c55e', 
                       "Optimal Density Field ŷ", 
                       "(60, 120) in [0, 1]\nContinuous Topology\nTruss Elements", '#86efac')
    draw_arrow((93.5, 40.5), (93.5, 36.5), color='#22c55e', lw=2)

    # ----------------------------------------------------
    # 7. Physics Loss Formula Box (Bottom Right Panel)
    # ----------------------------------------------------
    loss_box = FancyBboxPatch((76, 5), 22, 17, boxstyle="round,pad=0.4,rounding_size=0.6",
                             facecolor='#0f172a', edgecolor='#e11d48', linewidth=1.5, zorder=2)
    ax.add_patch(loss_box)
    ax.text(87, 19.8, "Physics-Guided Composite Loss", ha='center', va='center', 
            fontsize=9.5, fontweight='bold', color='#fda4af')
    ax.text(87, 16.8, r"$\mathcal{L}_{total} = \mathcal{L}_{BCE} + 0.5\mathcal{L}_{Dice} + \alpha\mathcal{L}_{Vol} + \beta\mathcal{L}_{Bin}$", 
            ha='center', va='center', fontsize=8.5, color='#ffffff')
    ax.text(87, 14.0, r"• $\mathcal{L}_{Vol} = (\mathrm{mean}(\hat{y}) - V_f)^2$  $(\alpha = 10.0)$", 
            ha='center', va='center', fontsize=8.0, color='#cbd5e1')
    ax.text(87, 11.5, r"• $\mathcal{L}_{Bin} = \frac{1}{N}\sum 4\hat{y}_e(1 - \hat{y}_e)$  $(\beta = 10^{-3})$", 
            ha='center', va='center', fontsize=8.0, color='#cbd5e1')
    ax.text(87, 9.0, r"• $\mathcal{L}_{Dice} = 1 - \frac{2\sum \hat{y} y + \epsilon}{\sum \hat{y} + \sum y + \epsilon}$", 
            ha='center', va='center', fontsize=8.0, color='#cbd5e1')
    ax.text(87, 6.8, "Enforces strict material limits & solid members", 
            ha='center', va='center', fontsize=7.2, color='#94a3b8', style='italic')

    # ----------------------------------------------------
    # 8. Legend / Metadata Banner (Bottom Left)
    # ----------------------------------------------------
    meta_box = FancyBboxPatch((2, 5), 26, 13, boxstyle="round,pad=0.3,rounding_size=0.5",
                              facecolor='#0f172a', edgecolor='#334155', linewidth=1.2, zorder=2)
    ax.add_patch(meta_box)
    ax.text(15, 15.5, "Architecture Specifications & Stats", ha='center', va='center', 
            fontsize=9.0, fontweight='bold', color='#38bdf8')
    specs = [
        "Total Parameters: 6,016,641 (6.02 M)",
        "Resolution: 60 x 120 Elements (14,762 DOFs)",
        "Inference Speed: 35.86 ms (CPU) | < 15 ms (GPU)",
        "DoubleConv: Conv2d(3x3) + BatchNorm + LeakyReLU(0.2)",
        "Optimizer: AdamW (lr=1e-3, CosineAnnealing)",
        "Dataset: 1,000 SIMP Solves (topo_dataset_1000.npz)"
    ]
    for idx, spec in enumerate(specs):
        ax.text(3.5, 13.2 - idx*1.7, f"✓ {spec}", ha='left', va='center', 
                fontsize=7.4, color='#cbd5e1', zorder=3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated U-Net Architecture Details diagram: {output_path}")

if __name__ == '__main__':
    create_unet_diagram()
