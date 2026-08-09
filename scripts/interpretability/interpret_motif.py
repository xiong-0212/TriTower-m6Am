#!/usr/bin/env python3

import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
import pandas as pd

import tritower_config as config

SEQ_LEN = config.SEQ_LEN
KMER_SIZE = 5
TOP_K_WINDOWS_PER_SAMPLE = 5
PSEUDO_COUNT = 1.0

IG_NPZ = os.path.join(config.PROJECT_ROOT, "output_interpretability", "ig_attributions.npz")
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


def load_attributions():
    if not os.path.exists(IG_NPZ):
        raise FileNotFoundError(
            f"Attribution file not found: {IG_NPZ}\n"
            f"Please run interpret_integrated_grad.py first to generate it."
        )
    data = np.load(IG_NPZ, allow_pickle=True)
    attributions = data["attributions"]
    labels = data["labels"]
    sequences = data["sequences"]
    return attributions, labels, sequences


def sequences_to_indices(seqs):
    nuc_to_idx = {"A": 0, "U": 1, "G": 2, "C": 3}
    n = len(seqs)
    idx_mat = np.zeros((n, SEQ_LEN), dtype=np.int32)
    for i, seq in enumerate(seqs):
        for j, ch in enumerate(seq):
            idx_mat[i, j] = nuc_to_idx.get(ch, 0)
    return idx_mat


def find_top_k_windows(max_attr, k=TOP_K_WINDOWS_PER_SAMPLE):
    window_scores = []
    for start in range(SEQ_LEN - KMER_SIZE + 1):
        score = max_attr[start:start + KMER_SIZE].sum()
        window_scores.append((start, score))
    window_scores.sort(key=lambda x: -x[1])
    return window_scores[:k]


def extract_high_attribution_kmers(sequences_idx, max_attr, labels, positive=True):
    mask = labels == 1 if positive else labels == 0
    kmer_list = []
    kmer_strings = []
    idx_to_nuc = ["A", "U", "G", "C"]

    for idx in range(len(labels)):
        if not mask[idx]:
            continue
        top_windows = find_top_k_windows(max_attr[idx], k=TOP_K_WINDOWS_PER_SAMPLE)
        for start_pos, _ in top_windows:
            kmer_indices = sequences_idx[idx, start_pos:start_pos + KMER_SIZE]
            kmer_list.append((start_pos, kmer_indices.copy()))
            kmer_str = "".join([idx_to_nuc[i] for i in kmer_indices])
            kmer_strings.append(kmer_str)

    return kmer_list, kmer_strings


def build_pwm(kmer_list):
    if not kmer_list:
        return None
    kmer_arr = np.array([k[1] for k in kmer_list])
    pwm = np.zeros((KMER_SIZE, 4), dtype=float)
    for i in range(KMER_SIZE):
        for nuc in range(4):
            pwm[i, nuc] = (kmer_arr[:, i] == nuc).sum()
    row_sums = pwm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    pwm = pwm / row_sums
    pwm_df = pd.DataFrame(pwm, columns=["A", "U", "G", "C"])
    return pwm_df


def compute_differential_kmers(pos_kmer_strings, neg_kmer_strings):
    pos_counter = Counter(pos_kmer_strings)
    neg_counter = Counter(neg_kmer_strings)
    all_kmers = set(pos_counter.keys()) | set(neg_counter.keys())

    n_pos = len(pos_kmer_strings)
    n_neg = len(neg_kmer_strings)

    results = []
    for kmer in all_kmers:
        pos_freq = (pos_counter.get(kmer, 0) + PSEUDO_COUNT) / (n_pos + PSEUDO_COUNT * len(all_kmers))
        neg_freq = (neg_counter.get(kmer, 0) + PSEUDO_COUNT) / (n_neg + PSEUDO_COUNT * len(all_kmers))
        log2fc = np.log2(pos_freq / neg_freq)
        results.append((kmer, pos_counter.get(kmer, 0), neg_counter.get(kmer, 0), log2fc))

    results.sort(key=lambda x: -x[3])
    return results


def compute_gc_content(kmer_strings):
    if not kmer_strings:
        return 0.0
    gc_counts = [sum(1 for c in k if c in 'GC') for k in kmer_strings]
    return float(np.mean(gc_counts) / KMER_SIZE)


def plot_sequence_logo(pwm_df, out_dir, name_prefix, color_scheme="classic"):
    if pwm_df is None:
        print(f"  [SKIP] No PWM data for {name_prefix}")
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    logomaker.Logo(pwm_df, ax=ax, color_scheme=color_scheme)
    ax.set_xlabel("Position in 5-mer", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xticks(range(KMER_SIZE))
    ax.set_xticklabels([str(i + 1) for i in range(KMER_SIZE)])
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{name_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{name_prefix}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_differential_kmers(diff_results, out_dir, top_n=15):
    top_enriched = diff_results[:top_n]
    top_depleted = diff_results[-top_n:][::-1]
    combined = top_enriched + top_depleted

    kmers = [r[0] for r in combined]
    log2fcs = [r[3] for r in combined]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in log2fcs]

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(kmers))
    ax.barh(y_pos, log2fcs, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(kmers, fontsize=9, fontfamily='monospace')
    ax.invert_yaxis()
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_xlabel("log2 Fold Change (Positive / Negative)", fontsize=11)
    ax.set_ylabel("5-mer Motif", fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_motif_differential.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_motif_differential.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_motif_differential.png/pdf")


def plot_window_position_dist(pos_kmer_list, neg_kmer_list, out_dir):
    pos_starts = [k[0] for k in pos_kmer_list]
    neg_starts = [k[0] for k in neg_kmer_list]

    bins = np.arange(-0.5, SEQ_LEN - KMER_SIZE + 1.5, 1)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(pos_starts, bins=bins, alpha=0.6, color="#d62728",
            label=f"Positive (n={len(pos_starts)})", density=True)
    ax.hist(neg_starts, bins=bins, alpha=0.6, color="#1f77b4",
            label=f"Negative (n={len(neg_starts)})", density=True)

    ax.axvline(x=22 - KMER_SIZE // 2, color="#d62728", linestyle="--",
               linewidth=1.5, alpha=0.7, label="IG pos peak (22)")
    ax.axvline(x=19 - KMER_SIZE // 2, color="#1f77b4", linestyle="--",
               linewidth=1.5, alpha=0.7, label="IG neg peak (19)")

    ax.set_xlabel("High-Attribution Window Start Position", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.set_xlim(-0.5, SEQ_LEN - KMER_SIZE + 0.5)
    ax.set_xticks(range(0, SEQ_LEN - KMER_SIZE + 1, 5))

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_window_position_dist.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_window_position_dist.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_window_position_dist.png/pdf")


def main():
    print("=" * 70)
    print("  TriTower-m6Am Interpretability: Sequence Motif Differential Analysis")
    print("=" * 70)

    print("\n[1/5] Loading IG attributions...")
    attributions, labels, sequences = load_attributions()
    print(f"  Loaded {len(labels)} samples: {int(labels.sum())} positive, "
          f"{int(len(labels) - labels.sum())} negative")

    sequences_idx = sequences_to_indices(sequences)

    max_attr = attributions.max(axis=2)

    print("\n[2/5] Extracting high-attribution 5-mers...")
    pos_kmer_list, pos_kmer_strings = extract_high_attribution_kmers(
        sequences_idx, max_attr, labels, positive=True
    )
    neg_kmer_list, neg_kmer_strings = extract_high_attribution_kmers(
        sequences_idx, max_attr, labels, positive=False
    )
    print(f"  Extracted {len(pos_kmer_strings)} 5-mers from positive samples")
    print(f"  Extracted {len(neg_kmer_strings)} 5-mers from negative samples")

    pos_pwm = build_pwm(pos_kmer_list)
    neg_pwm = build_pwm(neg_kmer_list)

    print("\n[3/5] Computing differential k-mer enrichment (log2 fold change)...")
    diff_results = compute_differential_kmers(pos_kmer_strings, neg_kmer_strings)

    top_enriched = diff_results[:10]
    top_depleted = diff_results[-10:][::-1]

    print("\n  Top-10 enriched in positive (log2FC > 0):")
    for kmer, pc, nc, fc in top_enriched:
        print(f"    {kmer}: pos={pc}, neg={nc}, log2FC={fc:+.3f}")

    print("\n  Top-10 depleted in positive (log2FC < 0):")
    for kmer, pc, nc, fc in top_depleted:
        print(f"    {kmer}: pos={pc}, neg={nc}, log2FC={fc:+.3f}")

    pos_gc = compute_gc_content(pos_kmer_strings)
    neg_gc = compute_gc_content(neg_kmer_strings)
    print(f"\n  GC content of extracted 5-mers:")
    print(f"    Positive: {pos_gc:.4f}")
    print(f"    Negative: {neg_gc:.4f}")
    print(f"    Difference: {pos_gc - neg_gc:+.4f}")

    print("\n[4/5] Analyzing high-attribution window positions...")
    pos_starts = np.array([k[0] for k in pos_kmer_list])
    neg_starts = np.array([k[0] for k in neg_kmer_list])
    print(f"  Positive window start: mean={pos_starts.mean():.2f}, median={np.median(pos_starts):.1f}")
    print(f"  Negative window start: mean={neg_starts.mean():.2f}, median={np.median(neg_starts):.1f}")
    print(f"  Difference (pos - neg): {pos_starts.mean() - neg_starts.mean():+.2f}")

    print("\n[5/5] Generating figures...")
    plot_sequence_logo(pos_pwm, OUT_DIR, "fig_motif_logo_pos", color_scheme="classic")
    print("  Saved: fig_motif_logo_pos.png/pdf")

    plot_sequence_logo(neg_pwm, OUT_DIR, "fig_motif_logo_neg", color_scheme="classic")
    print("  Saved: fig_motif_logo_neg.png/pdf")

    plot_differential_kmers(diff_results, OUT_DIR, top_n=15)
    plot_window_position_dist(pos_kmer_list, neg_kmer_list, OUT_DIR)

    summary = {
        "n_positive_kmers": len(pos_kmer_strings),
        "n_negative_kmers": len(neg_kmer_strings),
        "kmer_size": KMER_SIZE,
        "top_k_windows_per_sample": TOP_K_WINDOWS_PER_SAMPLE,
        "top10_enriched_in_positive": [
            {"kmer": k, "pos_count": pc, "neg_count": nc, "log2fc": float(fc)}
            for k, pc, nc, fc in top_enriched
        ],
        "top10_depleted_in_positive": [
            {"kmer": k, "pos_count": pc, "neg_count": nc, "log2fc": float(fc)}
            for k, pc, nc, fc in top_depleted
        ],
        "gc_content": {
            "positive": float(pos_gc),
            "negative": float(neg_gc),
            "difference": float(pos_gc - neg_gc),
        },
        "window_position_stats": {
            "positive_mean": float(pos_starts.mean()),
            "positive_median": float(np.median(pos_starts)),
            "negative_mean": float(neg_starts.mean()),
            "negative_median": float(np.median(neg_starts)),
            "difference": float(pos_starts.mean() - neg_starts.mean()),
        },
        "note": (
            "BCA motif enrichment analysis is NOT included because the dataset "
            "is constructed with the BCA constraint at positions 18-20 (pos 20 "
            "is always 'A' in ALL samples), so BCA has no discriminative power."
        ),
    }

    json_path = os.path.join(OUT_DIR, "motif_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    print("\n" + "=" * 70)
    print(f"  Done! Output directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
