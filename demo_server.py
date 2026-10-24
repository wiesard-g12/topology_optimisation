"""
demo_server.py - Interactive Browser-Based Web Demo for TopoUNet (Phase 2)
Serves a real-time web application using Python's standard library http.server (Zero extra dependencies).

Features:
- Live Canvas rendering of 60x120 structural domain with fixtures and loads
- Real-time U-Net inference vs SIMP Ground Truth side-by-side comparison
- Interactive controls: Target Volume Fraction Vf slider, Binarization threshold slider
- Presets: Cantilever Beam, MBB Beam, Bridge Truss, Crane, plus Dataset Samples (0-999)
- On-demand SIMP solver with memory caching
- Latency speedometer, Dice overlap, and volume parity analytics

Usage:
  python demo_server.py --port 5000 --checkpoint checkpoints/best_unet.pth --dataset topo_dataset_1000.npz
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
from dataset_generator import precompute_mesh_and_filter, solve_simp

# Global caches
_MODEL = None
_DEVICE = None
_DATASET = None
_SIMP_MATS = None
_SIMP_CACHE = {}


def get_model(checkpoint_path="checkpoints/best_unet.pth"):
    global _MODEL, _DEVICE
    if _MODEL is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _MODEL = TopoUNet(in_channels=6, out_channels=1)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=_DEVICE)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            try:
                _MODEL.load_state_dict(state_dict)
                print(f"Loaded TopoUNet checkpoint from {checkpoint_path} onto {_DEVICE}")
            except Exception as e:
                print(f"Notice: Loading checkpoint with strict=False: {e}")
                _MODEL.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Running initialized model.")
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _DEVICE


def get_dataset(dataset_path=None):
    global _DATASET
    if _DATASET is None:
        candidates = [dataset_path] if dataset_path else ["topo_dataset_1000.npz", "topo_dataset.npz"]
        for p in candidates:
            if p and os.path.exists(p):
                _DATASET = np.load(p)
                print(f"Loaded dataset {p} with {len(_DATASET['inputs'])} samples.")
                break
    return _DATASET


def get_simp_mats(nelx=120, nely=60, rmin=1.5):
    global _SIMP_MATS
    if _SIMP_MATS is None:
        print("Precomputing SIMP mesh and filter matrices for fast evaluation...")
        _SIMP_MATS = precompute_mesh_and_filter(nelx, nely, rmin)
    return _SIMP_MATS


def compute_or_get_simp(preset, volfrac, custom_idx=None):
    """
    Returns (simp_density, solve_time_ms, was_cached)
    """
    # 1. Dataset sample preset
    if preset.startswith('sample_') or custom_idx is not None:
        try:
            s_idx = custom_idx if custom_idx is not None else int(preset.split('_')[1])
            ds = get_dataset()
            if ds is not None and s_idx < len(ds['targets']):
                return ds['targets'][s_idx], 0.1, True
        except Exception as e:
            print(f"Dataset retrieval error: {e}")
        return None, 0.0, False

    # 2. Benchmark structural case: check memory cache
    cache_key = (preset, round(float(volfrac), 2))
    if cache_key in _SIMP_CACHE:
        return _SIMP_CACHE[cache_key], 0.1, True

    # 3. Solve SIMP on-the-fly (max_iter=25 gives crisp topology in ~2.5s)
    try:
        t0 = time.perf_counter()
        _, prob_meta = build_benchmark_problem(preset, volfrac=volfrac)
        edofMat, iK, jK, H, Hs = get_simp_mats(120, 60, 1.5)
        sol = solve_simp(120, 60, volfrac, 3.0, prob_meta['fixed_dofs'], prob_meta['F'],
                         edofMat, iK, jK, H, Hs, max_iter=25)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if sol is not None:
            _SIMP_CACHE[cache_key] = sol
            print(f"Solved SIMP for {preset} (Vf={volfrac:.2f}) in {dt_ms/1000.0:.2f}s")
            return sol, dt_ms, False
    except Exception as e:
        print(f"Error solving SIMP: {e}")
    return None, 0.0, False


def generate_plot_base64(input_tensor, pred_density, threshold=0.35, simp_density=None, 
                         latency_ms=0.0, simp_time_ms=0.0):
    nelx = input_tensor.shape[2]
    nely = input_tensor.shape[1]

    if input_tensor.shape[0] == 6:
        ch_bc = np.clip(input_tensor[1] + input_tensor[2], 0.0, 1.0)
        ch_fx = input_tensor[3]
        ch_fy = input_tensor[4]
        ch_vf = input_tensor[5]
    else:
        ch_bc = input_tensor[1]
        ch_fx = input_tensor[2]
        ch_fy = input_tensor[3]
        ch_vf = input_tensor[4]

    target_vf = float(np.mean(ch_vf))
    pred_vf = float(np.mean(pred_density))

    bin_pred = (pred_density >= threshold).astype(float)
    bin_vf = float(np.mean(bin_pred))

    n_cols = 4 if simp_density is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 3.6), dpi=140)

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
    ax1.set_title(rf"2. PI-UNet Continuous ($\hat{{y}}$)" + "\n" + f"Latency: {latency_ms:.1f} ms | $V_f = {pred_vf:.3f}$",
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

    # Panel 4: Ground Truth SIMP
    dice = None
    if simp_density is not None:
        bin_simp = (simp_density >= 0.5).astype(float)
        inter = np.sum(bin_pred * bin_simp)
        dice = float((2.0 * inter + 1e-6) / (np.sum(bin_pred) + np.sum(bin_simp) + 1e-6))
        simp_vf = float(np.mean(simp_density))

        time_str = f"{simp_time_ms/1000.0:.1f}s" if simp_time_ms > 100 else "Cached"
        ax3 = axes[3]
        ax3.imshow(1.0 - simp_density, cmap='gray', vmin=0, vmax=1,
                   extent=[0, nelx, nely, 0], interpolation='nearest')
        ax3.set_title(f"4. Ground Truth SIMP\nDice = {dice:.3f} | $V_f = {simp_vf:.3f}$ ({time_str})",
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
    return base64.b64encode(buf.read()).decode('utf-8'), dice


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
  <div class="max-w-7xl mx-auto space-y-6">
    <!-- Header -->
    <header class="flex items-center justify-between border-b border-slate-800 pb-5">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <span>⚡ TopoUNet</span>
          <span class="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">Phase 2 Demo</span>
          <span class="text-xs px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">SIMP Comparison Enabled</span>
        </h1>
        <p class="text-sm text-slate-400 mt-1">Physics-Induced Deep Learning Topology Optimization vs Iterative SIMP FEM Solver</p>
      </div>
      <div class="flex items-center gap-4 text-xs font-mono">
        <div class="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-right">
          <div class="text-slate-400">Resolution</div>
          <div class="text-white font-semibold">60 x 120 (14,762 DOFs)</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-right">
          <div class="text-slate-400">Parameters</div>
          <div class="text-emerald-400 font-semibold">8,744,153</div>
        </div>
      </div>
    </header>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Controls Column -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Design Specifications</h2>

        <!-- Preset Selector -->
        <div>
          <label class="block text-xs font-medium text-slate-300 mb-1.5">Structural Problem Preset</label>
          <select id="presetSelect" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500">
            <optgroup label="Standard Structural Cases">
              <option value="cantilever">Cantilever Beam (2 Left Pins, Tip Load)</option>
              <option value="mbb">MBB Half-Beam (Symmetry Roller, Corner Load)</option>
              <option value="bridge">Bridge Truss (Bottom Fixtures, Top Load)</option>
              <option value="crane">Crane / L-Bracket (Left Pins, Bottom Load)</option>
            </optgroup>
            <optgroup label="Verified Dataset Samples (1,000 Points)">
              <option value="sample_0">Dataset Sample #000 (Cantilever Style)</option>
              <option value="sample_1" selected>Dataset Sample #001 (Slender Truss - Dice 0.872)</option>
              <option value="sample_2">Dataset Sample #002 (Double Pin Support)</option>
              <option value="sample_3">Dataset Sample #003 (Triangular Web - Dice 0.801)</option>
              <option value="sample_4">Dataset Sample #004 (Complex Truss)</option>
              <option value="sample_custom">Custom Dataset Sample # (enter below)</option>
            </optgroup>
          </select>
        </div>

        <!-- Custom Sample ID Input (Hidden unless selected) -->
        <div id="customSampleDiv" class="hidden flex gap-2 items-center bg-slate-950/70 p-2.5 rounded-lg border border-slate-800">
          <label class="text-xs text-slate-400 whitespace-nowrap">Sample Index (0-999):</label>
          <input id="sampleIdxInput" type="number" min="0" max="999" value="1"
                 class="w-20 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white text-center font-mono">
          <button onclick="runInference()" class="text-xs bg-slate-700 hover:bg-slate-600 px-3 py-1 rounded text-white">Load</button>
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
            <span id="threshDisplay" class="font-mono text-purple-400 font-semibold">0.35</span>
          </div>
          <input id="threshSlider" type="range" min="0.10" max="0.90" step="0.05" value="0.35"
                 class="w-full accent-purple-500 bg-slate-800 rounded-lg h-2 cursor-pointer">
        </div>

        <!-- SIMP Ground Truth Option -->
        <div class="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 flex items-center justify-between">
          <label for="checkSimp" class="flex items-center gap-2.5 text-xs text-amber-200 cursor-pointer font-medium">
            <input type="checkbox" id="checkSimp" checked class="w-4 h-4 accent-amber-500 rounded cursor-pointer">
            <span>Show SIMP Calculated Ground Truth</span>
          </label>
          <span class="text-[10px] bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded font-mono">Panel 4</span>
        </div>

        <!-- Action Buttons -->
        <div class="pt-1 space-y-2">
          <button id="btnPredict" onclick="runInference()"
                  class="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white font-medium py-2.5 px-4 rounded-lg transition flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20">
            <span>⚡ Instant Predict (PI-UNet)</span>
          </button>
        </div>

        <!-- Performance Metrics Box -->
        <div class="border border-slate-800 rounded-lg p-3.5 bg-slate-950/50 space-y-2 text-xs">
          <div class="font-semibold text-slate-300 mb-1">Performance & Accuracy Scorecard</div>
          <div class="flex justify-between">
            <span class="text-slate-400">PI-UNet Latency:</span>
            <span id="metricLatency" class="font-mono text-emerald-400 font-semibold">-- ms</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">SIMP Solve Time:</span>
            <span id="metricSimpTime" class="font-mono text-amber-400 font-semibold">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Dice Similarity (vs SIMP):</span>
            <span id="metricDice" class="font-mono text-purple-400 font-bold">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Predicted Volume:</span>
            <span id="metricPredVf" class="font-mono text-slate-200">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Volume Error:</span>
            <span id="metricVfErr" class="font-mono text-slate-200">--</span>
          </div>
          <div class="flex justify-between border-t border-slate-800 pt-1.5 mt-1">
            <span class="text-slate-400">Speedup vs SIMP:</span>
            <span id="metricSpeedup" class="font-mono text-blue-400 font-bold">~250x faster</span>
          </div>
        </div>
      </div>

      <!-- Visualization Column -->
      <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Prediction & SIMP Comparison Output</h2>
            <div id="statusBadge" class="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-400">Ready</div>
          </div>

          <!-- Image Container -->
          <div class="bg-slate-950 border border-slate-800 rounded-lg p-3 min-h-[340px] flex items-center justify-center overflow-hidden">
            <img id="resultImg" class="max-w-full h-auto rounded transition duration-200" alt="Topology Output">
            <div id="spinner" class="hidden text-slate-400 text-sm animate-pulse flex flex-col items-center gap-2">
              <span>Running Deep Learning & SIMP Verification...</span>
            </div>
          </div>
        </div>

        <div class="mt-4 text-xs text-slate-400 flex justify-between items-center border-t border-slate-800 pt-3">
          <span>Resolution: 120 x 60 elements (14,762 DOFs)</span>
          <span class="text-slate-400">Architecture: PI-UNet (6-to-12 Ch Stem, FiLM ResBlock, Load Path MHA, Sigmund OC Filter)</span>
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
    const checkSimp = document.getElementById('checkSimp');
    const customDiv = document.getElementById('customSampleDiv');
    const sampleInput = document.getElementById('sampleIdxInput');

    vfSlider.addEventListener('input', (e) => { vfDisplay.innerText = Number(e.target.value).toFixed(2); });
    threshSlider.addEventListener('input', (e) => { threshDisplay.innerText = Number(e.target.value).toFixed(2); });

    presetSelect.addEventListener('change', () => {
      if (presetSelect.value === 'sample_custom') {
        customDiv.classList.remove('hidden');
      } else {
        customDiv.classList.add('hidden');
        runInference();
      }
    });

    checkSimp.addEventListener('change', () => {
      runInference();
    });

    async function runInference() {
      const btn = document.getElementById('btnPredict');
      const badge = document.getElementById('statusBadge');
      const spinner = document.getElementById('spinner');
      const img = document.getElementById('resultImg');

      btn.disabled = true;
      badge.innerText = checkSimp.checked ? 'Predicting & Verifying SIMP...' : 'Predicting...';
      badge.className = 'text-xs px-2.5 py-1 rounded bg-blue-500/20 text-blue-400';
      spinner.classList.remove('hidden');
      img.classList.add('opacity-30');

      let currentPreset = presetSelect.value;
      let customIdx = null;
      if (currentPreset === 'sample_custom') {
        customIdx = parseInt(sampleInput.value) || 0;
        currentPreset = 'sample_' + customIdx;
      }

      const payload = {
        preset: currentPreset,
        volfrac: parseFloat(vfSlider.value),
        threshold: parseFloat(threshSlider.value),
        compare_simp: checkSimp.checked,
        custom_idx: customIdx
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

        const targetVf = (data.target_vf !== undefined) ? data.target_vf : payload.volfrac;
        if (currentPreset.startsWith('sample_') || customIdx !== null) {
          vfSlider.value = targetVf.toFixed(2);
          vfDisplay.innerText = targetVf.toFixed(2);
        }
        document.getElementById('metricLatency').innerText = data.latency_ms.toFixed(1) + ' ms';
        document.getElementById('metricPredVf').innerText = data.predicted_vf.toFixed(3);
        document.getElementById('metricVfErr').innerText = Math.abs(data.predicted_vf - targetVf).toFixed(4);

        if (data.dice_score !== null && data.dice_score !== undefined) {
          const diceVal = data.dice_score.toFixed(3);
          const diceEl = document.getElementById('metricDice');
          diceEl.innerText = diceVal;
          diceEl.className = data.dice_score >= 0.70 ? 'font-mono text-emerald-400 font-bold' : 'font-mono text-amber-400 font-bold';
        } else {
          document.getElementById('metricDice').innerText = 'N/A';
        }

        if (data.simp_time_ms > 0) {
          const sSec = (data.simp_time_ms / 1000.0).toFixed(2) + ' s';
          document.getElementById('metricSimpTime').innerText = data.simp_cached ? 'Instant (Cached)' : sSec;
          const speedup = Math.round(data.simp_time_ms / Math.max(0.1, data.latency_ms));
          document.getElementById('metricSpeedup').innerText = speedup > 1 ? `~${speedup}x faster` : 'N/A';
        } else {
          document.getElementById('metricSimpTime').innerText = 'Instant (Dataset)';
        }

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

    // Run on initial load
    window.addEventListener('load', () => {
      runInference();
    });
  </script>
</body>
</html>
"""


class TopoDemoHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet clean server logs
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
            threshold = float(req.get('threshold', 0.35))
            compare_simp = bool(req.get('compare_simp', True))
            custom_idx = req.get('custom_idx', None)

            model, device = get_model()
            simp_density = None
            simp_time_ms = 0.0
            simp_cached = False

            if preset.startswith('sample_') or custom_idx is not None:
                s_idx = custom_idx if custom_idx is not None else int(preset.split('_')[1])
                ds = get_dataset()
                if ds is not None and s_idx < len(ds['inputs']):
                    input_tensor = ds['inputs'][s_idx]
                    if input_tensor.shape[0] == 5:
                        input_tensor = np.concatenate([input_tensor[0:1], input_tensor[1:2], input_tensor[1:2], input_tensor[2:5]], axis=0)
                    if compare_simp:
                        simp_density = ds['targets'][s_idx]
                    volfrac = float(np.mean(input_tensor[-1]))
                else:
                    input_tensor, _ = build_benchmark_problem('cantilever', volfrac=volfrac)
            else:
                input_tensor, _ = build_benchmark_problem(preset, volfrac=volfrac)
                if compare_simp:
                    simp_density, simp_time_ms, simp_cached = compute_or_get_simp(preset, volfrac)

            tensor_x = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                pred_tensor = model(tensor_x)
            pred_density = pred_tensor.squeeze().cpu().numpy()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            img_b64, dice = generate_plot_base64(
                input_tensor=input_tensor,
                pred_density=pred_density,
                threshold=threshold,
                simp_density=simp_density,
                latency_ms=latency_ms,
                simp_time_ms=simp_time_ms
            )

            resp = {
                'latency_ms': latency_ms,
                'target_vf': float(volfrac),
                'predicted_vf': float(np.mean(pred_density)),
                'dice_score': dice,
                'simp_time_ms': simp_time_ms,
                'simp_cached': simp_cached,
                'has_simp': simp_density is not None,
                'image_base64': img_b64
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port=5000, checkpoint="checkpoints/best_unet.pth", dataset=None):
    get_model(checkpoint)
    get_dataset(dataset)
    # Pre-warm SIMP filter matrices in background
    get_simp_mats(120, 60, 1.5)

    server_address = ('127.0.0.1', port)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, TopoDemoHandler) as httpd:
        print("=" * 60)
        print(f"  [FAST] TopoUNet / PI-UNet Interactive Web Demo Running!")
        print(f"  URL: http://127.0.0.1:{port}")
        print(f"  SIMP Ground Truth comparison: ENABLED")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


def main():
    parser = argparse.ArgumentParser(description="Run TopoUNet Web Demo")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_unet_rl.pth", help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, default="topo_dataset_3000.npz", help="Dataset path")
    args = parser.parse_args()
    start_server(port=args.port, checkpoint=args.checkpoint, dataset=args.dataset)


if __name__ == '__main__':
    main()
