#!/usr/bin/env python3

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tritower_config as config
from tritower_models import StructureRGCNTower
from struct_features import build_structural_node_features, clean_dot_bracket

CENTER_POS = 20
EDGE_TYPES = ["backbone", "pairing", "neighbor"]

CKPT_PATH = os.path.join(
    config.PROJECT_ROOT, "output_tritower", "ckpt_seed123", "rgcn_fold0.pth"
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


def build_graph(seq, ss, device="cpu"):
    from torch_geometric.data import Data
    from struct_features import parse_base_pairs

    n = len(seq)
    ss_clean = clean_dot_bracket(ss)
    node_feat = build_structural_node_features(seq, ss_clean).to(device)

    src, dst, etypes = [], [], []

    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        etypes.extend([0, 0])

    pairs = parse_base_pairs(ss_clean)
    for i, j in pairs.items():
        if i < j:
            src.extend([i, j])
            dst.extend([j, i])
            etypes.extend([1, 1])

    for i in range(n - 2):
        src.extend([i, i + 2])
        dst.extend([i + 2, i])
        etypes.extend([2, 2])

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_type = torch.tensor(etypes, dtype=torch.long, device=device)

    data = Data(x=node_feat, edge_index=edge_index, edge_type=edge_type)
    data.center_indices = torch.tensor([CENTER_POS], dtype=torch.long, device=device)
    data.batch = torch.zeros(n, dtype=torch.long, device=device)
    return data


def build_graph_missing_edge_type(seq, ss, missing_type, device="cpu"):
    from torch_geometric.data import Data
    from struct_features import parse_base_pairs

    n = len(seq)
    ss_clean = clean_dot_bracket(ss)
    node_feat = build_structural_node_features(seq, ss_clean).to(device)

    src, dst, etypes = [], [], []

    if missing_type != "backbone":
        for i in range(n - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])
            etypes.extend([0, 0])

    if missing_type != "pairing":
        pairs = parse_base_pairs(ss_clean)
        for i, j in pairs.items():
            if i < j:
                src.extend([i, j])
                dst.extend([j, i])
                etypes.extend([1, 1])

    if missing_type != "neighbor":
        for i in range(n - 2):
            src.extend([i, i + 2])
            dst.extend([i + 2, i])
            etypes.extend([2, 2])

    if len(src) == 0:
        edge_index = torch.tensor([[], []], dtype=torch.long, device=device)
        edge_type = torch.tensor([], dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_type = torch.tensor(etypes, dtype=torch.long, device=device)

    data = Data(x=node_feat, edge_index=edge_index, edge_type=edge_type)
    data.center_indices = torch.tensor([CENTER_POS], dtype=torch.long, device=device)
    data.batch = torch.zeros(n, dtype=torch.long, device=device)
    return data


def load_model(ckpt_path, device):
    model = StructureRGCNTower(hidden_dim=64, dropout=0.3)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def compute_edge_contribution(model, structures, labels, device):
    n_samples = len(structures)
    contributions = np.zeros((n_samples, 3), dtype=np.float32)
    original_probs = np.zeros(n_samples, dtype=np.float32)

    with torch.no_grad():
        for i, (seq, ss) in enumerate(structures):
            g_orig = build_graph(seq, ss, device)
            logits = model(g_orig)
            orig_prob = torch.sigmoid(logits).item()
            original_probs[i] = orig_prob

            for j, edge_type in enumerate(EDGE_TYPES):
                g_masked = build_graph_missing_edge_type(seq, ss, edge_type, device)
                logits_masked = model(g_masked)
                masked_prob = torch.sigmoid(logits_masked).item()
                contributions[i, j] = orig_prob - masked_prob

            if (i + 1) % 20 == 0 or (i + 1) == n_samples:
                print(f"    Processed {i + 1}/{n_samples} samples", flush=True)

    return contributions, original_probs


def plot_pos_neg_comparison(contributions, labels, out_dir):
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_means = contributions[pos_mask].mean(axis=0)
    neg_means = contributions[neg_mask].mean(axis=0)
    pos_stds = contributions[pos_mask].std(axis=0)
    neg_stds = contributions[neg_mask].std(axis=0)

    x = np.arange(len(EDGE_TYPES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars_pos = ax.bar(x - width / 2, pos_means, width, yerr=pos_stds,
                      color="#d62728", alpha=0.8, capsize=4,
                      label="Positive (m6Am)")
    bars_neg = ax.bar(x + width / 2, neg_means, width, yerr=neg_stds,
                      color="#1f77b4", alpha=0.8, capsize=4,
                      label="Negative")

    for bar in bars_pos:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.002 if h >= 0 else -0.002
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:.3f}",
                ha="center", va=va, fontsize=9)
    for bar in bars_neg:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.002 if h >= 0 else -0.002
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:.3f}",
                ha="center", va=va, fontsize=9)

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_ylabel("Contribution (Original - Masked)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in EDGE_TYPES], fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_pos_neg.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_pos_neg.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_edge_contribution_pos_neg.png/pdf")


def plot_violin_comparison(contributions, labels, out_dir):
    pos_mask = labels == 1
    neg_mask = labels == 0

    fig, ax = plt.subplots(figsize=(9, 5))

    positions = []
    data_list = []
    colors = []
    edge_labels_pos = []

    for i, etype in enumerate(EDGE_TYPES):
        pos_pos = i * 3 - 0.5
        neg_pos = i * 3 + 0.5
        positions.append(pos_pos)
        positions.append(neg_pos)
        data_list.append(contributions[pos_mask, i])
        data_list.append(contributions[neg_mask, i])
        colors.append("#d62728")
        colors.append("#1f77b4")
        edge_labels_pos.append(i * 3)

    parts = ax.violinplot(data_list, positions=positions, widths=1.0,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)

    rng = np.random.RandomState(42)
    for i, (pos, data, color) in enumerate(zip(positions, data_list, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(data))
        ax.scatter(pos + jitter, data, color=color, alpha=0.25, s=12,
                   edgecolor='none', zorder=2)

        median = np.median(data)
        ax.scatter(pos, median, color='white', edgecolor='black', s=50,
                   zorder=3, linewidths=1.2)

        mean = data.mean()
        if i % 2 == 0:
            text_x = pos - 0.7
            ha = "right"
        else:
            text_x = pos + 0.7
            ha = "left"
        ax.text(text_x, mean, f"μ={mean:+.3f}", ha=ha, va="center",
                fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, linewidth=0.8, alpha=0.9))

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xticks(edge_labels_pos)
    ax.set_xticklabels([e.capitalize() for e in EDGE_TYPES], fontsize=11)
    for i in range(len(EDGE_TYPES)):
        ax.text(i * 3 - 0.5, ax.get_ylim()[0] - 0.015, "Pos", ha="center",
                va="top", fontsize=9, color="#d62728", fontweight="bold")
        ax.text(i * 3 + 0.5, ax.get_ylim()[0] - 0.015, "Neg", ha="center",
                va="top", fontsize=9, color="#1f77b4", fontweight="bold")

    ax.set_ylabel("Contribution (Original - Masked)", fontsize=12)
    ax.set_xlim(-1.5, (len(EDGE_TYPES) - 1) * 3 + 1.5)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.04, y_max)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", alpha=0.6, label="Positive (m6Am)"),
        Patch(facecolor="#1f77b4", alpha=0.6, label="Negative"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, framealpha=0.9, loc="upper right")

    direction_notes = []
    pos_means_arr = contributions[pos_mask].mean(axis=0)
    neg_means_arr = contributions[neg_mask].mean(axis=0)
    for i, etype in enumerate(EDGE_TYPES):
        opposite = (pos_means_arr[i] > 0) != (neg_means_arr[i] > 0)
        note = f"{'↔' if opposite else '→'}"
        direction_notes.append((i * 3, note))
    for x_pos, note in direction_notes:
        ax.text(x_pos, ax.get_ylim()[1] - 0.005, note, ha="center", va="top",
                fontsize=14, fontweight="bold", color="#2c3e50")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_violin.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_violin.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_edge_contribution_violin.png/pdf")


def plot_grouped_bar_inbar(contributions, labels, out_dir):
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_means = contributions[pos_mask].mean(axis=0)
    neg_means = contributions[neg_mask].mean(axis=0)
    pos_stds = contributions[pos_mask].std(axis=0)
    neg_stds = contributions[neg_mask].std(axis=0)

    x = np.arange(len(EDGE_TYPES))
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_pos = ax.bar(x - width / 2, pos_means, width,
                      yerr=pos_stds, error_kw={"elinewidth": 0.8, "capsize": 2, "alpha": 0.6},
                      color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.8,
                      label="Positive (m6Am)")
    bars_neg = ax.bar(x + width / 2, neg_means, width,
                      yerr=neg_stds, error_kw={"elinewidth": 0.8, "capsize": 2, "alpha": 0.6},
                      color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.8,
                      label="Negative")

    for bar in list(bars_pos) + list(bars_neg):
        h = bar.get_height()
        if h >= 0:
            text_y = h / 2
            va = "center"
            color = "white"
        else:
            text_y = h / 2
            va = "center"
            color = "white"
        ax.text(bar.get_x() + bar.get_width() / 2, text_y,
                f"{h:+.3f}", ha="center", va=va,
                fontsize=10, fontweight="bold", color=color)

    for i in range(len(EDGE_TYPES)):
        opposite = (pos_means[i] > 0) != (neg_means[i] > 0)
        y_top = max(pos_means[i] + pos_stds[i], neg_means[i] + neg_stds[i])
        ax.annotate(
            "↕ Opposite" if opposite else "→ Same",
            xy=(i, y_top + 0.015),
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color="#2c3e50" if opposite else "#7f7f7f",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#2c3e50" if opposite else "#7f7f7f",
                      linewidth=0.8, alpha=0.9)
        )

    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.set_ylabel("Contribution (Original - Masked)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in EDGE_TYPES], fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")

    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_grouped_bar.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_grouped_bar.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_edge_contribution_grouped_bar.png/pdf")


def plot_faceted_subplots(contributions, labels, out_dir):
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_means = contributions[pos_mask].mean(axis=0)
    neg_means = contributions[neg_mask].mean(axis=0)
    pos_stds = contributions[pos_mask].std(axis=0)
    neg_stds = contributions[neg_mask].std(axis=0)

    y_max = max(pos_means.max() + pos_stds.max(),
                neg_means.max() + neg_stds.max(),
                abs(pos_means.min() - pos_stds.min()),
                abs(neg_means.min() - neg_stds.min())) * 1.35

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), sharey=True)

    bar_width = 0.45

    for i, (ax, etype) in enumerate(zip(axes, EDGE_TYPES)):
        x_pos = 0
        x_neg = 1
        bars_pos = ax.bar(x_pos, pos_means[i], bar_width,
                          yerr=pos_stds[i],
                          color="#d62728", alpha=0.85,
                          edgecolor="black", linewidth=0.8,
                          error_kw={"elinewidth": 1.0, "capsize": 5, "alpha": 0.7},
                          label="Positive (m6Am)")
        bars_neg = ax.bar(x_neg, neg_means[i], bar_width,
                          yerr=neg_stds[i],
                          color="#1f77b4", alpha=0.85,
                          edgecolor="black", linewidth=0.8,
                          error_kw={"elinewidth": 1.0, "capsize": 5, "alpha": 0.7},
                          label="Negative")

        for bars, val in [(bars_pos, pos_means[i]), (bars_neg, neg_means[i])]:
            bar = bars[0] if hasattr(bars, '__getitem__') else bars
            h = bar.get_height()
            if abs(h) > 0.001:
                text_y = h / 2
                ax.text(bar.get_x() + bar.get_width() / 2, text_y,
                        f"{val:+.4f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")

        opposite = (pos_means[i] > 0) != (neg_means[i] > 0)
        if opposite:
            arrow_text = "↕ Opposite"
            arrow_color = "#cc6600"
        else:
            arrow_text = "→ Same"
            arrow_color = "#7f7f7f"

        ax.annotate(arrow_text,
                    xy=(0.5, y_max * 0.92),
                    ha="center", va="top",
                    fontsize=10, fontweight="bold", color=arrow_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=arrow_color, linewidth=0.8, alpha=0.95))

        ax.axhline(y=0, color="black", linewidth=1.0)

        ax.set_xticks([x_pos, x_neg])
        ax.set_xticklabels(["Positive", "Negative"], fontsize=10)

        ax.set_title(etype.capitalize(), fontsize=12, fontweight="bold", pad=10)

        ax.set_ylim(-y_max, y_max)

        ax.yaxis.grid(True, linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Contribution (Original - Masked)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=10, framealpha=0.95)

    fig.suptitle("Edge Type Contribution: Positive vs Negative Samples",
                 fontsize=13, fontweight="bold", y=1.08)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_faceted.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_faceted.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_edge_contribution_faceted.png/pdf")


def plot_radar_comparison(contributions, labels, out_dir):
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_means = contributions[pos_mask].mean(axis=0)
    neg_means = contributions[neg_mask].mean(axis=0)
    pos_stds = contributions[pos_mask].std(axis=0)
    neg_stds = contributions[neg_mask].std(axis=0)

    N = len(EDGE_TYPES)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    pos_values = np.concatenate([pos_means, [pos_means[0]]])
    neg_values = np.concatenate([neg_means, [neg_means[0]]])

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1.1], wspace=0.20,
                          left=0.04, right=0.97, top=0.80, bottom=0.12)

    ax = fig.add_subplot(gs[0, 0], polar=True)

    max_abs = max(abs(pos_means).max(), abs(neg_means).max()) * 1.25
    ax.set_ylim(-max_abs, max_abs)

    ax.plot(angles, pos_values, color="#d62728", linewidth=2.5,
            label="Positive (m6Am)", zorder=3)
    ax.fill(angles, pos_values, color="#d62728", alpha=0.25, zorder=2)

    ax.plot(angles, neg_values, color="#1f77b4", linewidth=2.5,
            label="Negative", zorder=3)
    ax.fill(angles, neg_values, color="#1f77b4", alpha=0.25, zorder=2)

    ax.plot(angles, [0] * len(angles), color="gray", linewidth=1.0,
            linestyle="--", alpha=0.7, zorder=1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([e.capitalize() for e in EDGE_TYPES],
                       fontsize=12, fontweight="bold")
    ax.tick_params(axis='x', which='major', pad=18)

    from matplotlib.transforms import ScaledTranslation
    _dx = 10 / 72
    _offset = ScaledTranslation(_dx, 0, fig.dpi_scale_trans)
    _xlabels = ax.get_xticklabels()
    if len(_xlabels) > 0:
        _xlabels[0].set_transform(_xlabels[0].get_transform() + _offset)

    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='major', pad=0)

    ax.text(np.pi / 6, 0, "0", fontsize=8, color="gray", alpha=0.6,
            ha="center", va="center")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15),
              fontsize=10, framealpha=0.95, ncol=2)

    ax_table = fig.add_subplot(gs[0, 1])
    ax_table.axis("off")

    col_labels = ["Edge type", "Pos (mean ± SD)", "Neg (mean ± SD)", "Direction"]
    rows = []
    for i, etype in enumerate(EDGE_TYPES):
        opposite = (pos_means[i] > 0) != (neg_means[i] > 0)
        direction = "↔ Opposite" if opposite else "→ Same"
        rows.append([
            etype.capitalize(),
            f"{pos_means[i]:+.4f} ± {pos_stds[i]:.4f}",
            f"{neg_means[i]:+.4f} ± {neg_stds[i]:.4f}",
            direction,
        ])

    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.20, 0.30, 0.30, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 3.0)

    n_rows = len(rows) + 1
    for r in range(n_rows):
        for c in range(len(col_labels)):
            cell = table[r, c]
            if r == 0:
                cell.set_facecolor("#36454f")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                if c == 0:
                    cell.set_text_props(fontweight="bold")
                if c == 1 and r > 0:
                    cell.set_facecolor("#fde8e8")
                    cell.set_text_props(color="#d62728", fontweight="bold")
                elif c == 2 and r > 0:
                    cell.set_facecolor("#e8f0fd")
                    cell.set_text_props(color="#1f77b4", fontweight="bold")
                elif c == 3 and r > 0:
                    if "Opposite" in rows[r - 1][3]:
                        cell.set_facecolor("#fff4e6")
                        cell.set_text_props(color="#cc6600", fontweight="bold")
                    else:
                        cell.set_facecolor("#f0f0f0")
                        cell.set_text_props(color="#666666")
                cell.set_edgecolor("#cccccc")
                cell.set_linewidth(0.5)

    fig.suptitle("Edge Type Contribution: Positive vs Negative Samples",
                 fontsize=13, fontweight="bold", y=0.95)

    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_radar.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_edge_contribution_radar.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_edge_contribution_radar.png/pdf")


def main():
    print("=" * 70)
    print("  TriTower-m6Am Interpretability: RGCN Edge Type Contribution")
    print("  Method: Occlusion Sensitivity (perturbation-based)")
    print("=" * 70)

    device = config.DEVICE
    print(f"\n  Device: {device}")

    print("\n[1/5] Loading test dataset...")
    test_structures, test_labels = config.load_structures_and_labels(
        os.path.join(config.DATA_DIR, "test_ss_41.fasta"), 320, 320
    )
    test_labels = np.array(test_labels, dtype=np.float32)
    print(f"  Test samples: {len(test_labels)}")

    print("\n[2/5] Loading trained RGCN tower checkpoint...")
    if not os.path.exists(CKPT_PATH):
        print(f"  ERROR: Checkpoint not found at {CKPT_PATH}")
        return
    model = load_model(CKPT_PATH, device)
    print(f"  Loaded checkpoint: {CKPT_PATH}")

    print("\n[3/5] Computing edge type contributions via occlusion...")
    contributions, original_probs = compute_edge_contribution(
        model, test_structures, test_labels, device
    )
    print(f"  Contributions shape: {contributions.shape}")

    print("\n[4/5] Generating figures...")
    print("  [a] Original bar chart...")
    plot_pos_neg_comparison(contributions, test_labels, OUT_DIR)
    print("  [b] Violin + scatter overlay...")
    plot_violin_comparison(contributions, test_labels, OUT_DIR)
    print("  [c] Grouped bar (in-bar labels)...")
    plot_grouped_bar_inbar(contributions, test_labels, OUT_DIR)
    print("  [d] Faceted subplots (one panel per edge type)...")
    plot_faceted_subplots(contributions, test_labels, OUT_DIR)
    print("  [e] Radar chart + side table...")
    plot_radar_comparison(contributions, test_labels, OUT_DIR)

    print("\n[5/5] Saving results...")
    npz_path = os.path.join(OUT_DIR, "edge_contribution_results.npz")
    np.savez(
        npz_path,
        contributions=contributions,
        original_probs=original_probs,
        labels=test_labels,
        edge_types=np.array(EDGE_TYPES),
    )
    print(f"  Saved: {npz_path}")

    pos_mask = test_labels == 1
    neg_mask = test_labels == 0
    pos_means = contributions[pos_mask].mean(axis=0)
    neg_means = contributions[neg_mask].mean(axis=0)

    summary = {
        "edge_types": EDGE_TYPES,
        "positive_means": pos_means.tolist(),
        "negative_means": neg_means.tolist(),
        "positive_stds": contributions[pos_mask].std(axis=0).tolist(),
        "negative_stds": contributions[neg_mask].std(axis=0).tolist(),
        "direction_consistency": {
            EDGE_TYPES[i]: {
                "pos_direction": "positive" if pos_means[i] > 0 else "negative",
                "neg_direction": "positive" if neg_means[i] > 0 else "negative",
                "opposite": bool((pos_means[i] > 0) != (neg_means[i] > 0)),
            }
            for i in range(3)
        },
    }

    print("\n  --- Key Findings ---")
    for i, etype in enumerate(EDGE_TYPES):
        print(f"  {etype}: pos={pos_means[i]:+.4f}, neg={neg_means[i]:+.4f}, "
              f"opposite={'YES' if (pos_means[i] > 0) != (neg_means[i] > 0) else 'NO'}")

    json_path = os.path.join(OUT_DIR, "edge_contribution_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    print("\n" + "=" * 70)
    print(f"  Done! Output directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
