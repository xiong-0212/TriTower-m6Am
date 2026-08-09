#!/usr/bin/env python3

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tritower_config as config

TOWER_NAMES = ["rnafm", "onehot", "rgcn"]
DEFAULT_THRESHOLD = 0.47
N_FOLDS = 5
SEED = 123

TRAIN_OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "output_tritower")
CKPT_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"ckpt_seed{SEED}")
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


def load_tower_predictions():
    test_structures, test_labels = config.load_structures_and_labels(
        os.path.join(config.DATA_DIR, "test_ss_41.fasta"), 320, 320
    )
    test_labels = np.array(test_labels, dtype=np.float32)

    tower_probs = {tower: np.zeros(len(test_labels), dtype=np.float32) for tower in TOWER_NAMES}
    n_fold_loaded = 0
    for fold_idx in range(N_FOLDS):
        fold_npz = os.path.join(CKPT_DIR, f"fold{fold_idx}.npz")
        if not os.path.exists(fold_npz):
            print(f"  [WARN] {fold_npz} not found, skipping fold {fold_idx}")
            continue
        data = np.load(fold_npz, allow_pickle=True)
        for tower in TOWER_NAMES:
            key = f"test_{tower}"
            if key not in data:
                raise KeyError(f"Key '{key}' not found in {fold_npz}. Available: {data.files}")
            tower_probs[tower] += data[key] / N_FOLDS
        n_fold_loaded += 1
        print(f"  Loaded fold {fold_idx}: {fold_npz}")

    if n_fold_loaded == 0:
        raise FileNotFoundError(
            f"No fold npz files found in {CKPT_DIR}. "
            f"Expected fold0.npz ... fold{N_FOLDS-1}.npz. "
            f"Please run train_tritower.py first."
        )
    if n_fold_loaded < N_FOLDS:
        print(f"  [WARN] Only {n_fold_loaded}/{N_FOLDS} folds loaded. Renormalizing predictions.")
        for tower in TOWER_NAMES:
            tower_probs[tower] = tower_probs[tower] * (N_FOLDS / n_fold_loaded)

    for tower in TOWER_NAMES:
        print(f"  {tower}: shape={tower_probs[tower].shape}, "
              f"range=[{tower_probs[tower].min():.4f}, {tower_probs[tower].max():.4f}]")

    return tower_probs, test_labels


def load_ensemble_weights():
    json_path = os.path.join(TRAIN_OUTPUT_DIR, "final_result.json")
    if not os.path.exists(json_path):
        print(f"  [WARN] {json_path} not found, using equal weights [1/3, 1/3, 1/3]")
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    with open(json_path, 'r') as f:
        result = json.load(f)

    weights_key = "ensemble_weights" if "ensemble_weights" in result else "weights"
    if weights_key in result:
        w = result[weights_key]
        weights = np.array([w["rnafm"], w["onehot"], w["rgcn"]])
        print(f"  Loaded ensemble weights from '{weights_key}': "
              f"rnafm={weights[0]:.4f}, onehot={weights[1]:.4f}, rgcn={weights[2]:.4f}")
        return weights
    else:
        print(f"  [WARN] No ensemble_weights in final_result.json, using equal weights")
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])


def load_threshold():
    json_path = os.path.join(TRAIN_OUTPUT_DIR, "final_result.json")
    if not os.path.exists(json_path):
        print(f"  [INFO] final_result.json not found, using default threshold={DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD
    try:
        with open(json_path, 'r') as f:
            result = json.load(f)
        thresh = float(result["final"]["thresh"])
        print(f"  Loaded threshold from final_result.json: {thresh:.4f}")
        return thresh
    except (KeyError, ValueError, TypeError):
        print(f"  [WARN] Could not parse threshold from final_result.json, using default={DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD


def compute_tower_contributions(tower_probs, weights):
    n = len(next(iter(tower_probs.values())))
    contributions = np.zeros((n, 3), dtype=np.float32)
    for i, tower in enumerate(TOWER_NAMES):
        contributions[:, i] = weights[i] * tower_probs[tower]

    ensemble_probs = contributions.sum(axis=1)
    return contributions, ensemble_probs


def classify_predictions(ensemble_probs, labels, threshold):
    pred = (ensemble_probs >= threshold).astype(int)
    tp_mask = (pred == 1) & (labels == 1)
    fp_mask = (pred == 1) & (labels == 0)
    fn_mask = (pred == 0) & (labels == 1)
    tn_mask = (pred == 0) & (labels == 0)
    return {
        "TP": tp_mask, "FP": fp_mask, "FN": fn_mask, "TN": tn_mask
    }


def plot_violin(tower_probs, labels, ensemble_probs, threshold, out_dir):
    categories = classify_predictions(ensemble_probs, labels, threshold)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, tower in enumerate(TOWER_NAMES):
        ax = axes[i]
        data_list = []
        labels_list = []
        for cat_name, mask in categories.items():
            if mask.sum() > 0:
                data_list.append(tower_probs[tower][mask])
                labels_list.append(f"{cat_name}\n(n={mask.sum()})")

        if data_list:
            parts = ax.violinplot(data_list, positions=range(len(data_list)),
                                  showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(data_list)))
            ax.set_xticklabels(labels_list, fontsize=9)

        ax.set_title(tower, fontsize=12)
        ax.set_ylabel("Prediction Probability" if i == 0 else "", fontsize=11)
        ax.axhline(y=threshold, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_tower_contribution_violin.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_tower_contribution_violin.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_tower_contribution_violin.png/pdf")


def plot_confusion_scatter(tower_probs, labels, ensemble_probs, threshold, out_dir):
    categories = classify_predictions(ensemble_probs, labels, threshold)

    fig, ax = plt.subplots(figsize=(7, 6))

    cat_colors = {
        "TP": "#d62728", "FP": "#ff9896",
        "FN": "#1f77b4", "TN": "#aec7e8"
    }

    for cat_name, mask in categories.items():
        if mask.sum() > 0:
            ax.scatter(tower_probs["rnafm"][mask], tower_probs["rgcn"][mask],
                       c=cat_colors[cat_name], label=f"{cat_name} (n={mask.sum()})",
                       alpha=0.6, s=20, edgecolors='none')

    ax.set_xlabel("RNA-FM Tower Probability", fontsize=12)
    ax.set_ylabel("RGCN Tower Probability", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_tower_confusion_scatter.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_tower_confusion_scatter.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_tower_confusion_scatter.png/pdf")


def main():
    print("=" * 70)
    print("  TriTower-m6Am Interpretability: Per-Sample Tower Contribution")
    print("=" * 70)

    print("\n[1/6] Loading per-tower predictions...")
    tower_probs, test_labels = load_tower_predictions()
    print(f"  Test samples: {len(test_labels)}")

    print("\n[2/6] Loading ensemble weights...")
    weights = load_ensemble_weights()
    for i, tower in enumerate(TOWER_NAMES):
        print(f"  {tower}: weight={weights[i]:.4f}")

    print("\n[3/6] Loading decision threshold...")
    threshold = load_threshold()
    print(f"  Using threshold: {threshold:.4f}")

    print("\n[4/6] Computing per-sample tower contributions...")
    contributions, ensemble_probs = compute_tower_contributions(tower_probs, weights)
    print(f"  Ensemble prob range: [{ensemble_probs.min():.4f}, {ensemble_probs.max():.4f}]")

    print("\n[5/6] Generating figures...")
    plot_violin(tower_probs, test_labels, ensemble_probs, threshold, OUT_DIR)
    plot_confusion_scatter(tower_probs, test_labels, ensemble_probs, threshold, OUT_DIR)

    print("\n[6/6] Saving results...")
    npz_path = os.path.join(OUT_DIR, "tower_contribution_results.npz")
    np.savez(
        npz_path,
        rnafm_probs=tower_probs["rnafm"],
        onehot_probs=tower_probs["onehot"],
        rgcn_probs=tower_probs["rgcn"],
        ensemble_probs=ensemble_probs,
        labels=test_labels,
        weights=weights,
        threshold=threshold,
        contributions=contributions,
    )
    print(f"  Saved: {npz_path}")

    categories = classify_predictions(ensemble_probs, test_labels, threshold)
    cat_stats = {}
    for cat_name, mask in categories.items():
        if mask.sum() > 0:
            cat_stats[cat_name] = {
                "n": int(mask.sum()),
                "rnafm_mean": float(tower_probs["rnafm"][mask].mean()),
                "onehot_mean": float(tower_probs["onehot"][mask].mean()),
                "rgcn_mean": float(tower_probs["rgcn"][mask].mean()),
                "ensemble_mean": float(ensemble_probs[mask].mean()),
            }

    summary = {
        "ensemble_weights": {"rnafm": float(weights[0]), "onehot": float(weights[1]), "rgcn": float(weights[2])},
        "threshold": float(threshold),
        "category_stats": cat_stats,
    }

    print("\n  --- Category Stats ---")
    for cat, stats in cat_stats.items():
        print(f"  {cat} (n={stats['n']}): "
              f"rnafm={stats['rnafm_mean']:.3f}, "
              f"onehot={stats['onehot_mean']:.3f}, "
              f"rgcn={stats['rgcn_mean']:.3f}")

    json_path = os.path.join(OUT_DIR, "tower_contribution_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    print("\n" + "=" * 70)
    print(f"  Done! Output directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
