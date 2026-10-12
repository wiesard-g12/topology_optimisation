"""
demo_server.py - Interactive Browser-Based Web Demo for TopoUNet (Phase 2)
Serves a real-time web application using Python's standard library http.server (Zero extra dependencies).

Features:
- Live Canvas rendering of 60x120 structural domain with fixtures and loads
- Interactive controls: Target Volume Fraction Vf slider, Binarization threshold slider
- Presets: Cantilever Beam, MBB Beam, Bridge Truss, Crane, Dataset Samples
- Real-time U-Net inference via REST API: returns density arrays & base64 rendered topologies
- Latency speedometer & compliance analysis

Usage:
  python demo_server.py --port 5000
  Open http://localhost:5000 in your browser
"""

import http.server
import socketserver
import json
import os
import io
import time
import base64
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from unet_model import TopoUNet
from interactive_demo import build_benchmark_problem
from visualize_data import extract_discrete_loads, draw_loads, draw_domain_and_bcs

# Global model and dataset cache
_MODEL = None
_DEVICE = None
_DATASET = None


def get_model(checkpoint_path="checkpoints/best_unet.pth"):
    global _MODEL, _DEVICE
    if _MODEL is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _MODEL = TopoUNet(in_channels=5, out_channels=1)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=_DEVICE)
            _MODEL.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            print(f"Loaded TopoUNet checkpoint from {checkpoint_path} onto {_DEVICE}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Running initialized model.")
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _DEVICE


def get_dataset(dataset_path="topo_dataset.npz"):
    global _DATASET
    if _DATASET is None and os.path.exists(dataset_path):
        _DATASET = np.load(dataset_path)
        print(f"Loaded dataset {dataset_path} with {len(_DATASET['inputs'])} samples.")
    return _DATASET


def generate_plot_base64(input_tensor, pred_density, threshold=0.5, simp_density=None, latency_ms=0.0):
    nelx = input_tensor.shape[2]
    nely = input_tensor.shape[1]
    ch_bc = input_tensor[1]
    ch_fx = input_tensor[2]
    ch_fy = input_tensor[3]
    ch_vf = input_tensor[4]
    target_vf = float(np.mean(ch_vf))
    pred_vf = float(np.mean(pred_density))
    
    bin_pred = (pred_density >= threshold).astype(float)
    bin_vf = float(np.mean(bin_pred))
    
    n_cols = 4 if simp_density is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 3.8), dpi=140)
    
    # Panel 1: BCs & Loads
    ax0 = axes[0]
    draw_domain_and_bcs(ax0, ch_bc)
    loads = extract_discrete_loads(ch_fx, ch_fy)
    draw_loads(ax0, loads)
    ax0.set_title(f"1. Problem Setup\nTarget $V_f = {target_vf:.2f}$", fontsize=10, fontweight='bold', pad=6)
    ax0.set_xlim(-15, nelx + 15)
    ax0.set_ylim(nely + 10, -10)
    ax0.set_aspect('equal')
    ax0.axis('off')
    
    # Panel 2: Continuous Pred
    ax1 = axes[1]
    ax1.imshow(1.0 - pred_density, cmap='gray', vmin=0, vmax=1,
               extent=[0, nelx, nely, 0], interpolation='nearest')
    ax1.set_title(rf"2. U-Net Continuous ($\hat{{y}}$)" + "\n" + f"Latency: {latency_ms:.1f} ms | $V_f = {pred_vf:.3f}$",
                  fontsize=10, fontweight='bold', pad=6)
    ax1.set_xlim(-5, nelx + 5)
    ax1.set_ylim(nely + 5, -5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Panel 3: Binarized
    ax2 = axes[2]
    ax2.imshow(1.0 - bin_pred, cmap='gray', vmin=0, vmax=1,
               extent=[0, nelx, nely, 0], interpolation='nearest')
    ax2.set_title(rf"3. Binarized Truss ($\hat{{y}} \geq {threshold:.2f}$)" + "\n" + f"$V_f = {bin_vf:.3f}$",
                  fontsize=10, fontweight='bold', pad=6)
    ax2.set_xlim(-5, nelx + 5)
    ax2.set_ylim(nely + 5, -5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Panel 4 (Optional): Ground Truth SIMP
    if simp_density is not None:
        bin_simp = (simp_density >= 0.5).astype(float)
        inter = np.sum(bin_pred * bin_simp)
        dice = (2.0 * inter + 1e-6) / (np.sum(bin_pred) + np.sum(bin_simp) + 1e-6)
        
        ax3 = axes[3]
        ax3.imshow(1.0 - simp_density, cmap='gray', vmin=0, vmax=1,
                   extent=[0, nelx, nely, 0], interpolation='nearest')
        ax3.set_title(f"4. Ground Truth SIMP\nDice = {dice:.3f} | $V_f = {float(np.mean(simp_density)):.3f}$",
                      fontsize=10, fontweight='bold', pad=6)
        ax3.set_xlim(-5, nelx + 5)
        ax3.set_ylim(nely + 5, -5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>TopoUNet: Real-Time Structural Topology Optimization</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-6">
  <div class="max-w-6xl mx-auto space-y-6">
    <!-- Header -->
    <header class="flex items-center justify-between border-b border-slate-800 pb-5">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <span>⚡ TopoUNet</span>
          <span class="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">Phase 2 Demo</span>
        </h1>
        <p class="text-sm text-slate-400 mt-1">Real-Time Deep Learning Structural Topology Optimization (< 100 ms Inference)</p>
      </div>
      <div class="flex items-center gap-4 text-xs font-mono">
        <div class="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-right">
          <div class="text-slate-400">Model Resolution</div>
          <div class="text-white font-semibold">60 x 120 Grid (Q4)</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-right">
          <div class="text-slate-400">Parameters</div>
          <div class="text-emerald-400 font-semibold">6,016,641</div>
        </div>
      </div>
    </header>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Controls Column -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-5">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Design Specifications</h2>

        <!-- Preset Selector -->
        <div>
          <label class="block text-xs font-medium text-slate-300 mb-1.5">Structural Problem Preset</label>
          <select id="presetSelect" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500">
            <option value="cantilever">Cantilever Beam (2 Left Pins, Tip Load)</option>
            <option value="mbb">MBB Half-Beam (Symmetry Roller, Corner Load)</option>
            <option value="bridge">Bridge Truss (Bottom Fixtures, Top Load)</option>
            <option value="crane">Crane / L-Bracket (Left Pins, Bottom Load)</option>
            <option value="sample_0">Dataset Sample #000</option>
            <option value="sample_1">Dataset Sample #001 (Dice 0.732)</option>
            <option value="sample_3">Dataset Sample #003 (Dice 0.816)</option>
          </select>
        </div>

        <!-- Volume Fraction Slider -->
        <div>
          <div class="flex justify-between text-xs font-medium text-slate-300 mb-1">
            <span>Target Volume Fraction (Vf)</span>
            <span id="vfDisplay" class="font-mono text-blue-400 font-semibold">0.40</span>
          </div>
          <input id="vfSlider" type="range" min="0.20" max="0.60" step="0.01" value="0.40"
                 class="w-full accent-blue-500 bg-slate-800 rounded-lg h-2 cursor-pointer">
          <div class="flex justify-between text-[11px] text-slate-500 mt-1">
            <span>0.20 (Lightweight)</span>
            <span>0.60 (Heavy Duty)</span>
          </div>
        </div>

        <!-- Binarization Threshold Slider -->
        <div>
          <div class="flex justify-between text-xs font-medium text-slate-300 mb-1">
            <span>Binarization Cutoff Threshold</span>
            <span id="threshDisplay" class="font-mono text-purple-400 font-semibold">0.50</span>
          </div>
          <input id="threshSlider" type="range" min="0.10" max="0.90" step="0.05" value="0.50"
                 class="w-full accent-purple-500 bg-slate-800 rounded-lg h-2 cursor-pointer">
        </div>

        <!-- Action Buttons -->
        <div class="pt-2 space-y-2.5">
          <button id="btnPredict" onclick="runInference()"
                  class="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white font-medium py-2.5 px-4 rounded-lg transition flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20">
            <span>⚡ Instant Predict (U-Net)</span>
          </button>
        </div>

        <!-- Performance Metrics Box -->
        <div class="border border-slate-800 rounded-lg p-3.5 bg-slate-950/50 space-y-2 text-xs">
          <div class="font-semibold text-slate-300 mb-1">Inference Performance</div>
          <div class="flex justify-between">
            <span class="text-slate-400">Latency:</span>
            <span id="metricLatency" class="font-mono text-emerald-400 font-semibold">-- ms</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Predicted Volume:</span>
            <span id="metricPredVf" class="font-mono text-slate-200">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Volume Error:</span>
            <span id="metricVfErr" class="font-mono text-slate-200">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Speedup vs SIMP:</span>
            <span id="metricSpeedup" class="font-mono text-blue-400 font-bold">~600x faster</span>
          </div>
        </div>
      </div>

      <!-- Visualization Column -->
      <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Prediction Output</h2>
            <div id="statusBadge" class="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-400">Ready</div>
          </div>

          <!-- Image Container -->
          <div class="bg-slate-950 border border-slate-800 rounded-lg p-3 min-h-[300px] flex items-center justify-center overflow-hidden">
            <img id="resultImg" class="max-w-full h-auto rounded transition duration-200" alt="Topology Output">
            <div id="spinner" class="hidden text-slate-400 text-sm animate-pulse">Running Neural Inference...</div>
          </div>
        </div>

        <div class="mt-4 text-xs text-slate-400 flex justify-between items-center border-t border-slate-800 pt-3">
          <span>Resolution: 120 x 60 elements (14,762 DOFs)</span>
          <span class="text-slate-400">Architecture: TopoUNet (4-level Encoder-Decoder)</span>
        </div>
      </div>
    </div>
  </div>

  <script>
    const vfSlider = document.getElementById('vfSlider');
    const vfDisplay = document.getElementById('vfDisplay');
    const threshSlider = document.getElementById('threshSlider');
    const threshDisplay = document.getElementById('threshDisplay');
    const presetSelect = document.getElementById('presetSelect');

    vfSlider.addEventListener('input', (e) => { vfDisplay.innerText = Number(e.target.value).toFixed(2); });
    threshSlider.addEventListener('input', (e) => { threshDisplay.innerText = Number(e.target.value).toFixed(2); });

    presetSelect.addEventListener('change', () => {
      runInference();
    });

    async function runInference() {
      const btn = document.getElementById('btnPredict');
      const badge = document.getElementById('statusBadge');
      const spinner = document.getElementById('spinner');
      const img = document.getElementById('resultImg');

      btn.disabled = true;
      badge.innerText = 'Predicting...';
      badge.className = 'text-xs px-2.5 py-1 rounded bg-blue-500/20 text-blue-400';
      spinner.classList.remove('hidden');
      img.classList.add('opacity-30');

      const payload = {
        preset: presetSelect.value,
        volfrac: parseFloat(vfSlider.value),
        threshold: parseFloat(threshSlider.value)
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        img.src = 'data:image/png;base64,' + data.image_base64;
        img.classList.remove('opacity-30');
        spinner.classList.add('hidden');

        document.getElementById('metricLatency').innerText = data.latency_ms.toFixed(1) + ' ms';
        document.getElementById('metricPredVf').innerText = data.predicted_vf.toFixed(3);
        document.getElementById('metricVfErr').innerText = Math.abs(data.predicted_vf - payload.volfrac).toFixed(4);

        badge.innerText = 'Completed in ' + data.latency_ms.toFixed(1) + ' ms';
        badge.className = 'text-xs px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-400';
      } catch (err) {
        badge.innerText = 'Error';
        badge.className = 'text-xs px-2.5 py-1 rounded bg-red-500/20 text-red-400';
        console.error(err);
      } finally {
        btn.disabled = false;
      }
    }

    // Run on load
    window.addEventListener('load', () => {
      runInference();
    });
  </script>
</body>
</html>
"""


class TopoDemoHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean logging
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode('utf-8'))
            
            preset = req.get('preset', 'cantilever')
            volfrac = float(req.get('volfrac', 0.40))
            threshold = float(req.get('threshold', 0.50))
            
            model, device = get_model()
            simp_density = None
            
            if preset.startswith('sample_'):
                sample_idx = int(preset.split('_')[1])
                ds = get_dataset()
                if ds is not None and sample_idx < len(ds['inputs']):
                    input_tensor = ds['inputs'][sample_idx]
                    simp_density = ds['targets'][sample_idx]
                    volfrac = float(np.mean(input_tensor[4]))
                else:
                    input_tensor, _ = build_benchmark_problem('cantilever', volfrac=volfrac)
            else:
                input_tensor, _ = build_benchmark_problem(preset, volfrac=volfrac)
                
            tensor_x = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
            
            t0 = time.perf_counter()
            with torch.no_grad():
                pred_tensor = model(tensor_x)
            pred_density = pred_tensor.squeeze().cpu().numpy()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            
            img_b64 = generate_plot_base64(
                input_tensor=input_tensor,
                pred_density=pred_density,
                threshold=threshold,
                simp_density=simp_density,
                latency_ms=latency_ms
            )
            
            resp = {
                'latency_ms': latency_ms,
                'predicted_vf': float(np.mean(pred_density)),
                'image_base64': img_b64
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port=5000):
    get_model()
    get_dataset()
    server_address = ('127.0.0.1', port)
    
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, TopoDemoHandler) as httpd:
        print("=" * 60)
        print(f"  [FAST] TopoUNet Interactive Web Demo Running!")
        print(f"  URL: http://127.0.0.1:{port}")
        print(f"  Press Ctrl+C to stop the server.")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


def main():
    parser = argparse.ArgumentParser(description="Run TopoUNet Web Demo")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    args = parser.parse_args()
    start_server(port=args.port)


if __name__ == '__main__':
    main()
