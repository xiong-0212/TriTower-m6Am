#!/usr/bin/env python3

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

import tritower_config as config
from tritower_models import KmerBiLSTMTower

SEQ_LEN = config.SEQ_LEN
NUC_MAP = {
    "A": [1, 0, 0, 0],
    "U": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "C": [0, 0, 0, 1],
}
CENTER_POS = 20

CKPT_PATH = os.path.join(
    config.PROJECT_ROOT, "output_tritower", "ckpt_seed123", "onehot_fold0.pth"
)
OUT_DIR = os.path.join(config.PROJECT_ROOT, "output_interpretability")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def build_onehot_matrix(seqs_list):
    n = len(seqs_list)
    mat = np.zeros((n, SEQ_LEN, 4), dtype=np.float32)
    for i, seq in enumerate(seqs_list):
        for j, ch in enumerate(seq):
            if ch in NUC_MAP:
                mat[i, j] = NUC_MAP[ch]
    return mat


def sequences_from_structures(structures):
    return [s[0].upper() for s in structures]


def load_model(ckpt_path, device):
    model = KmerBiLSTMTower(input_dim=4, hidden_dim=64, num_layers=2, dropout=0.3)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def compute_attributions(model, onehot_tensor, device, n_steps=50, batch_size=64):
    ig = IntegratedGradients(model)
    n_samples = onehot_tensor.size(0)
    all_attr = []
    all_probs = []

    original_state = model.training
    lstm_dropout_backup = {}
    model.train()
    for m in model.modules():
        if isinstance(m, nn.LSTM):
            lstm_dropout_backup[id(m)] = m.dropout
            m.dropout = 0.0
        elif isinstance(m, nn.Dropout):
            m.p_backup = m.p
            m.p = 0.0

    try:
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_input = onehot_tensor[start:end].to(device)
            batch_input.requires_grad = True

            attr = ig.attribute(batch_input, baselines=torch.zeros_like(batch_input),
                                n_steps=n_steps, target=None)
            all_attr.append(attr.detach().cpu().numpy())

            with torch.no_grad():
                logits = model(batch_input)
                probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    finally:
        for m in model.modules():
            if isinstance(m, nn.LSTM):
                m.dropout = lstm_dropout_backup.get(id(m), m.dropout)
            elif isinstance(m, nn.Dropout) and hasattr(m, 'p_backup'):
                m.p = m.p_backup
                del m.p_backup
        model.train(original_state)

    attributions = np.concatenate(all_attr, axis=0)
    predictions = np.concatenate(all_probs, axis=0)
    return attributions, predictions


def plot_saliency_map(attributions, predictions, labels, sequences, out_dir):
    max_attr = attributions.max(axis=2)

    confidence = np.abs(predictions - 0.5)
    sorted_indices = np.argsort(confidence)[::-1]
    top_k = min(50, len(sorted_indices))
    top_indices = sorted_indices[:top_k]

    heatmap_data = max_attr[top_indices]
    top_labels = labels[top_indices]

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = sns.color_palette("magma", as_cmap=True)
    im = ax.imshow(heatmap_data, aspect="auto", cmap=cmap, interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Max Attribution Score", fontsize=11)

    ax.set_xlabel("Sequence Position", fontsize=12)
    ax.set_ylabel("Sample (sorted by confidence)", fontsize=12)

    tick_positions = list(range(0, SEQ_LEN, 5))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(p) for p in tick_positions], fontsize=9)

    ax.set_yticks(range(0, top_k, 10))
    ax.set_yticklabels([f"#{i}" for i in range(0, top_k, 10)], fontsize=9)

    ax.axvline(x=CENTER_POS, color="cyan", linewidth=1.5, linestyle="--", alpha=0.8)

    for i in range(top_k):
        color = "#d62728" if top_labels[i] == 1 else "#1f77b4"
        ax.add_patch(plt.Rectangle((-0.8, i - 0.5), 0.6, 1, color=color, clip_on=False))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Positive (m6Am)"),
        Patch(facecolor="#1f77b4", label="Negative"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_ig_saliency_map.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_ig_saliency_map.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig A saved: fig_ig_saliency_map.png/pdf")


def plot_average_profile(attributions, labels, out_dir):
    max_attr = attributions.max(axis=2)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_attr = max_attr[pos_mask]
    neg_attr = max_attr[neg_mask]

    pos_mean = pos_attr.mean(axis=0)
    pos_std = pos_attr.std(axis=0)
    neg_mean = neg_attr.mean(axis=0)
    neg_std = neg_attr.std(axis=0)

    positions = np.arange(SEQ_LEN)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(positions, pos_mean, color="#d62728", linewidth=2, label="Positive (m6Am)")
    ax.fill_between(positions, pos_mean - pos_std, pos_mean + pos_std,
                    color="#d62728", alpha=0.2)

    ax.plot(positions, neg_mean, color="#1f77b4", linewidth=2, label="Negative")
    ax.fill_between(positions, neg_mean - neg_std, neg_mean + neg_std,
                    color="#1f77b4", alpha=0.2)

    ax.axvline(x=CENTER_POS, color="gray", linewidth=1.5, linestyle="--", alpha=0.7)

    ax.set_xlabel("Sequence Position", fontsize=12)
    ax.set_ylabel("Max Attribution Score", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(0, SEQ_LEN - 1)
    ax.set_xticks(range(0, SEQ_LEN, 5))

    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_ig_avg_profile.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_ig_avg_profile.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig B saved: fig_ig_avg_profile.png/pdf")


def plot_top_snippets(attributions, predictions, labels, sequences, out_dir):
    sample_importance = np.abs(attributions).sum(axis=(1, 2))
    top_indices = np.argsort(sample_importance)[::-1][:10]

    max_attr = attributions.max(axis=2)

    fig, axes = plt.subplots(10, 1, figsize=(14, 12), sharex=True)
    if len(top_indices) == 1:
        axes = [axes]

    nuc_colors = {"A": "#2ca02c", "U": "#d62728", "G": "#ff7f0e", "C": "#1f77b4"}

    for rank, idx in enumerate(top_indices):
        ax = axes[rank]
        seq = sequences[idx]
        attr_profile = max_attr[idx]

        bar_colors = [nuc_colors.get(seq[j], "#888888") for j in range(SEQ_LEN)]
        ax.bar(range(SEQ_LEN), attr_profile, color=bar_colors, alpha=0.7, width=0.9)

        top5_pos = np.argsort(attr_profile)[::-1][:5]
        for p in top5_pos:
            if attr_profile[p] > 0:
                ax.text(p, attr_profile[p] + attr_profile.max() * 0.05,
                        seq[p], ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color=nuc_colors.get(seq[p], "black"))

        ax.axvspan(18, 22, alpha=0.1, color="red", zorder=0)
        ax.axvline(x=CENTER_POS, color="red", linewidth=1, linestyle=":", alpha=0.5)

        ax.set_ylabel(f"#{rank+1}", fontsize=9, rotation=0, va="center")
        ax.set_ylim(0, attr_profile.max() * 1.3 if attr_profile.max() > 0 else 1)
        ax.tick_params(axis="y", labelsize=7)

    axes[-1].set_xlabel("Sequence Position", fontsize=12)
    axes[-1].set_xticks(range(0, SEQ_LEN, 2))
    axes[-1].set_xticklabels(range(0, SEQ_LEN, 2), fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n, alpha=0.7) for n, c in nuc_colors.items()]
    legend_elements.append(Patch(facecolor="red", alpha=0.1, label="BCA motif region"))
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8,
                   ncol=5, framealpha=0.9)

    plt.tight_layout(h_pad=0.5)
    fig.savefig(os.path.join(out_dir, "fig_ig_top_snippets.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_ig_top_snippets.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig C saved: fig_ig_top_snippets.png/pdf")


def main():
    print("=" * 70)
    print("  TriTower-m6Am Interpretability: Integrated Gradients")
    print("  One-hot BiLSTM Tower Nucleotide Attribution Analysis")
    print("=" * 70)

    device = config.DEVICE
    print(f"\n  Device: {device}")

    print("\n[1/5] Loading test dataset...")
    test_structures, test_labels = config.load_structures_and_labels(
        os.path.join(config.DATA_DIR, "test_ss_41.fasta"), 320, 320
    )
    test_labels = np.array(test_labels, dtype=np.float32)
    test_seqs = sequences_from_structures(test_structures)
    test_onehot = build_onehot_matrix(test_seqs)

    print(f"  Test samples: {len(test_labels)} "
          f"(pos={int(test_labels.sum())}, neg={int(len(test_labels) - test_labels.sum())})")
    print(f"  One-hot shape: {test_onehot.shape}")

    print("\n[2/5] Loading trained One-hot BiLSTM tower...")
    if not os.path.exists(CKPT_PATH):
        print(f"  ERROR: Checkpoint not found at {CKPT_PATH}")
        return
    model = load_model(CKPT_PATH, device)
    print(f"  Loaded checkpoint: {CKPT_PATH}")

    print("\n[3/5] Computing Integrated Gradients attributions (n_steps=50)...")
    onehot_tensor = torch.tensor(test_onehot, dtype=torch.float32)
    attributions, predictions = compute_attributions(
        model, onehot_tensor, device, n_steps=50, batch_size=64
    )
    print(f"  Attributions shape: {attributions.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Attribution range: [{attributions.min():.4f}, {attributions.max():.4f}]")

    print("\n[4/5] Generating figures...")
    plot_saliency_map(attributions, predictions, test_labels, test_seqs, OUT_DIR)
    plot_average_profile(attributions, test_labels, OUT_DIR)
    plot_top_snippets(attributions, predictions, test_labels, test_seqs, OUT_DIR)

    print("\n[5/5] Saving results...")

    npz_path = os.path.join(OUT_DIR, "ig_attributions.npz")
    np.savez(
        npz_path,
        attributions=attributions,
        max_attr_per_pos=attributions.max(axis=2),
        predictions=predictions,
        labels=test_labels,
        sequences=np.array(test_seqs),
    )
    print(f"  Saved: {npz_path}")

    max_attr = attributions.max(axis=2)
    pos_mask = test_labels == 1
    neg_mask = test_labels == 0
    pos_mean = max_attr[pos_mask].mean(axis=0)
    neg_mean = max_attr[neg_mask].mean(axis=0)

    summary = {
        "n_samples": int(len(test_labels)),
        "n_positive": int(test_labels.sum()),
        "n_negative": int(len(test_labels) - test_labels.sum()),
        "attribution_range": [float(attributions.min()), float(attributions.max())],
        "positive_peak_pos": int(np.argmax(pos_mean)),
        "positive_peak_val": float(pos_mean.max()),
        "negative_peak_pos": int(np.argmax(neg_mean)),
        "negative_peak_val": float(neg_mean.max()),
        "center_pos20_pos_mean": float(pos_mean[CENTER_POS]),
        "center_pos20_neg_mean": float(neg_mean[CENTER_POS]),
        "center_pos20_diff": float(pos_mean[CENTER_POS] - neg_mean[CENTER_POS]),
        "max_diff_pos": int(np.argmax(pos_mean - neg_mean)),
        "max_diff_val": float((pos_mean - neg_mean).max()),
        "pos_mean_per_position": pos_mean.tolist(),
        "neg_mean_per_position": neg_mean.tolist(),
    }

    print("\n  --- Key Findings ---")
    print(f"  Positive peak: pos {summary['positive_peak_pos']} (val={summary['positive_peak_val']:.4f})")
    print(f"  Negative peak: pos {summary['negative_peak_pos']} (val={summary['negative_peak_val']:.4f})")
    print(f"  Center (pos 20): pos={summary['center_pos20_pos_mean']:.4f}, neg={summary['center_pos20_neg_mean']:.4f}")
    print(f"  Max difference at pos {summary['max_diff_pos']}: {summary['max_diff_val']:.4f}")

    json_path = os.path.join(OUT_DIR, "ig_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 70)
    print(f"  Done! Output directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
