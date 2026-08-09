#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tritower_config as config
from tritower_models import RNAFMTower, BellPooling

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLOR_INIT = '#A8A8A8'
COLOR_TRAINED = '#D62728'

CKPT_PATH = os.path.join(
    config.PROJECT_ROOT, "output_tritower", "ckpt_seed123", "rnafm_fold0.pth"
)
OUT_DIR = os.path.join(config.PROJECT_ROOT, "output_interpretability")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("  BellPooling Interpretability Analysis")
    print("  TriTower-m6Am RNA-FM Tower")
    print("=" * 70)

    print("\n[1/4] Loading trained RNA-FM Tower checkpoint...")
    model = RNAFMTower(seq_len=config.SEQ_LEN).to(config.DEVICE)

    if os.path.exists(CKPT_PATH):
        state_dict = torch.load(CKPT_PATH, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Loaded checkpoint: {CKPT_PATH}")
    else:
        print(f"  ERROR: Checkpoint not found at {CKPT_PATH}")
        return

    model.eval()

    print("\n[2/4] Extracting BellPooling weights...")
    with torch.no_grad():
        trained_weights = model.bell().cpu().numpy()

    bell_init = BellPooling(seq_len=config.SEQ_LEN)
    with torch.no_grad():
        init_weights = bell_init().cpu().numpy()

    positions = np.arange(config.SEQ_LEN)
    center = config.SEQ_LEN // 2

    weight_diff = trained_weights - init_weights
    center_init = init_weights[center]
    center_trained = trained_weights[center]
    center_change = center_trained - center_init

    def safe_entropy(w):
        w = np.clip(w, 1e-12, 1.0)
        w = w / w.sum()
        return -np.sum(w * np.log(w))

    init_entropy = safe_entropy(init_weights)
    trained_entropy = safe_entropy(trained_weights)

    top5_idx = np.argsort(trained_weights)[::-1][:5]

    print(f"  Center (pos {center}) init weight:    {center_init:.6f}")
    print(f"  Center (pos {center}) trained weight: {center_trained:.6f}")
    print(f"  Center change: {center_change:+.6f}")
    print(f"  Init entropy:    {init_entropy:.4f}")
    print(f"  Trained entropy: {trained_entropy:.4f}")
    print(f"  Top-5 positions (by trained weight): {sorted(top5_idx.tolist())}")
    print(f"  Max weight position: {np.argmax(trained_weights)} (val={trained_weights.max():.6f})")
    print(f"  Min weight position: {np.argmin(trained_weights)} (val={trained_weights.min():.6f})")

    print("\n[3/4] Generating figure...")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    ax.plot(positions, init_weights, color=COLOR_INIT, linewidth=1.8,
            linestyle='--', marker='o', markersize=3, label='Initialized (bell-shaped prior)',
            alpha=0.8, zorder=2)
    ax.plot(positions, trained_weights, color=COLOR_TRAINED, linewidth=2.2,
            linestyle='-', marker='s', markersize=3, label='Trained',
            alpha=0.9, zorder=3)

    ax.axvline(x=center, color='#333333', linestyle=':', linewidth=1.0, alpha=0.5)

    ax.set_xlabel('Sequence Position', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.5, config.SEQ_LEN - 0.5)
    ax.set_xticks(positions[::5])
    ax.set_xticklabels(positions[::5])

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_bell_pooling_weights.png'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_bell_pooling_weights.pdf'))
    plt.close(fig)
    print("  [OK] fig_bell_pooling_weights.png/pdf")

    print("\n[4/4] Saving numerical results...")
    npz_path = os.path.join(OUT_DIR, "bell_pooling_results.npz")
    np.savez(
        npz_path,
        positions=positions,
        init_weights=init_weights,
        trained_weights=trained_weights,
        weight_diff=weight_diff,
        center_pos=center,
        center_init=center_init,
        center_trained=center_trained,
        center_change=center_change,
        init_entropy=init_entropy,
        trained_entropy=trained_entropy,
        top5_positions=top5_idx,
        max_pos=np.argmax(trained_weights),
        max_weight=trained_weights.max(),
        min_pos=np.argmin(trained_weights),
        min_weight=trained_weights.min(),
    )
    print(f"  Saved: {npz_path}")

    print("\n" + "=" * 70)
    print("  Summary for paper:")
    print(f"  - Center position weight: {center_init:.4f} -> {center_trained:.4f} (change={center_change:+.4f})")
    print(f"  - Entropy: {init_entropy:.4f} -> {trained_entropy:.4f}")
    print(f"    (lower = more concentrated)")
    print(f"  - Max weight at pos {np.argmax(trained_weights)} (val={trained_weights.max():.4f})")
    print(f"  - Peak shifted from center (pos {center}) to pos {np.argmax(trained_weights)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
