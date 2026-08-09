#!/usr/bin/env python3

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
import pandas as pd

import tritower_config as config

SEQ_LEN = config.SEQ_LEN
CENTER_POS = 20
BCA_REGION = (18, 22)
TOP_K_POSITIVE = 50

NUC_LIST = ["A", "U", "G", "C"]
NUC_TO_IDX = {"A": 0, "U": 1, "G": 2, "C": 3}

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


def load_ig_data():
    if not os.path.exists(IG_NPZ):
        raise FileNotFoundError(
            f"Attribution file not found: {IG_NPZ}\n"
            f"Please run interpret_integrated_grad.py first."
        )
    data = np.load(IG_NPZ, allow_pickle=True)
    return (data["attributions"], data["predictions"],
            data["labels"], data["sequences"])


def build_ig_weighted_pwm(sequences, attributions, labels, predictions,
                          top_k=TOP_K_POSITIVE):
    pos_mask = labels == 1
    pos_indices = np.where(pos_mask)[0]
    pos_conf = np.abs(predictions[pos_indices] - 0.5)
    sorted_order = np.argsort(pos_conf)[::-1]
    n_select = min(top_k, len(sorted_order))
    selected = pos_indices[sorted_order[:n_select]]

    max_attr = np.maximum(attributions.max(axis=2), 0.0)

    pwm = np.zeros((SEQ_LEN, 4), dtype=float)
    for idx in selected:
        seq = sequences[idx]
        for i, ch in enumerate(seq):
            nuc_idx = NUC_TO_IDX.get(ch, 0)
            pwm[i, nuc_idx] += max_attr[idx, i]

    row_sums = pwm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pwm = pwm / row_sums

    per_pos_weight = np.maximum(attributions[selected].max(axis=2), 0.0).sum(axis=0)

    pwm_df = pd.DataFrame(pwm, columns=NUC_LIST)
    return pwm_df, selected, per_pos_weight


def plot_ig_weighted_logo(pwm_df, per_pos_weight, out_dir,
                          zoom=False, name_prefix="fig_ig_weighted_logo"):
    if zoom:
        start, end = 15, 26
        pwm_plot = pwm_df.iloc[start:end].reset_index(drop=True)
        weight_plot = per_pos_weight[start:end]
        xticks = list(range(start, end))
        title = "IG-weighted motif (BCA region, pos 15-25)"
        figsize = (8, 3.5)
    else:
        pwm_plot = pwm_df.copy()
        weight_plot = per_pos_weight
        xticks = list(range(SEQ_LEN))
        title = "IG-weighted sequence logo (top-{} positive samples)".format(
            TOP_K_POSITIVE)
        figsize = (12, 4)

    fig, ax = plt.subplots(figsize=figsize)

    logomaker.Logo(pwm_plot, ax=ax,
                   color_scheme="classic",
                   stack_order="big_on_top",
                   vpad=0.05,
                   fade_probabilities=False)

    ax.set_xlabel("Sequence Position", fontsize=11)
    ax.set_ylabel("Information (bits)" if not zoom else "Frequency",
                  fontsize=11)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels([str(p) for p in xticks], fontsize=8)

    if zoom:
        bca_start_in_zoom = 18 - start
        bca_end_in_zoom = 22 - start
        ax.axvspan(bca_start_in_zoom - 0.5, bca_end_in_zoom - 0.5,
                   alpha=0.15, color="red", zorder=0, label="BCA motif")
        ax.axvline(x=CENTER_POS - start, color="red",
                   linewidth=1.2, linestyle="--", alpha=0.7)
    else:
        ax.axvspan(BCA_REGION[0] - 0.5, BCA_REGION[1] - 0.5,
                   alpha=0.15, color="red", zorder=0, label="BCA motif region")
        ax.axvline(x=CENTER_POS, color="red",
                   linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 2.0)

    plt.tight_layout()
    suffix = "_zoom" if zoom else ""
    png_path = os.path.join(out_dir, f"{name_prefix}{suffix}.png")
    pdf_path = os.path.join(out_dir, f"{name_prefix}{suffix}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(png_path)} / {os.path.basename(pdf_path)}")


def plot_per_position_weight(per_pos_weight, out_dir):
    fig, ax = plt.subplots(figsize=(12, 3))
    positions = np.arange(SEQ_LEN)
    ax.bar(positions, per_pos_weight, color="#5DADE2",
           alpha=0.8, edgecolor="white", linewidth=0.5)

    ax.axvspan(BCA_REGION[0] - 0.5, BCA_REGION[1] - 0.5,
               alpha=0.15, color="red", zorder=0, label="BCA motif region")
    ax.axvline(x=CENTER_POS, color="red", linewidth=1.2,
               linestyle="--", alpha=0.7)

    ax.set_xlabel("Sequence Position", fontsize=11)
    ax.set_ylabel("Total IG Weight", fontsize=11)
    ax.set_title("Total IG attribution weight per position (top-{} positives)".format(
        TOP_K_POSITIVE), fontsize=11)
    ax.set_xticks(range(0, SEQ_LEN, 5))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "fig_ig_per_position_weight.png")
    pdf_path = os.path.join(out_dir, "fig_ig_per_position_weight.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_ig_per_position_weight.png/pdf")


def main():
    print("=" * 70)
    print("  TriTower-m6Am Interpretability: IG-weighted Sequence Logo")
    print("=" * 70)

    print("\n[1/4] Loading IG attributions...")
    attributions, predictions, labels, sequences = load_ig_data()
    n_pos = int(labels.sum())
    print(f"  Loaded {len(labels)} samples ({n_pos} positive)")
    print(f"  Attributions shape: {attributions.shape}")

    print(f"\n[2/4] Building IG-weighted PWM from top-{TOP_K_POSITIVE} positives...")
    pwm_df, selected, per_pos_weight = build_ig_weighted_pwm(
        sequences, attributions, labels, predictions, top_k=TOP_K_POSITIVE
    )
    print(f"  Selected {len(selected)} top-confidence positive samples")
    print(f"  Per-position weight range: [{per_pos_weight.min():.4f}, "
          f"{per_pos_weight.max():.4f}]")
    print(f"  Peak position: {int(np.argmax(per_pos_weight))} "
          f"(weight={per_pos_weight.max():.4f})")

    bca_start, bca_end = BCA_REGION
    bca_weight = per_pos_weight[bca_start:bca_end].sum()
    total_weight = per_pos_weight.sum()
    bca_frac = bca_weight / total_weight if total_weight > 0 else 0.0
    print(f"\n  BCA region (pos {bca_start}-{bca_end-1}) weight: {bca_weight:.4f}")
    print(f"  Total weight: {total_weight:.4f}")
    print(f"  BCA fraction: {bca_frac:.4f}")

    pos20_nucs = [sequences[i][CENTER_POS] for i in selected]
    pos20_counts = {n: pos20_nucs.count(n) for n in NUC_LIST}
    print(f"  Pos 20 nucleotide composition (selected positives): {pos20_counts}")

    print("\n[3/4] Generating figures...")
    plot_ig_weighted_logo(pwm_df, per_pos_weight, OUT_DIR, zoom=False)
    plot_ig_weighted_logo(pwm_df, per_pos_weight, OUT_DIR, zoom=True)
    plot_per_position_weight(per_pos_weight, OUT_DIR)

    print("\n[4/4] Saving summary...")
    summary = {
        "n_selected_positive": int(len(selected)),
        "top_k": TOP_K_POSITIVE,
        "per_position_total_weight": per_pos_weight.tolist(),
        "peak_position": int(np.argmax(per_pos_weight)),
        "peak_weight": float(per_pos_weight.max()),
        "bca_region": {"start": bca_start, "end": bca_end},
        "bca_region_total_weight": float(bca_weight),
        "bca_region_fraction": float(bca_frac),
        "pos20_nucleotide_counts": pos20_counts,
        "pwm": pwm_df.values.tolist(),
        "interpretation": (
            "Letter heights in the logo reflect IG-attribution-weighted "
            "nucleotide frequencies. The BCA motif region (pos 18-22) is "
            "highlighted. Pos 20 is fixed to A by dataset construction; "
            "enrichment at neighboring positions reflects model attention "
            "rather than sequence variability."
        ),
    }
    json_path = os.path.join(OUT_DIR, "ig_weighted_logo_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {os.path.basename(json_path)}")

    print("\n" + "=" * 70)
    print(f"  Done! Output directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
