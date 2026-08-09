#!/usr/bin/env python3

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import tritower_config as config
from struct_features import build_structural_node_features

CENTER_POS = 20

OUT_DIR = os.path.join(config.PROJECT_ROOT, "output_interpretability")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "A", "U", "G", "C",
    "SS_OpenParen", "SS_CloseParen", "SS_Unpaired",
    "Loop_Paired", "Loop_Hairpin", "Loop_Internal", "Loop_Bulge",
    "Loop_Multi", "Loop_Exterior",
    "Pair_Distance", "Pair_Density", "Stacking", "Loop_Length",
    "Pos_Sin", "Pos_Cos", "Is_Paired", "GC_Content",
]

DEFAULT_RGCN_AUC = 0.7644

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


def load_rgcn_auc():
    json_path = os.path.join(config.PROJECT_ROOT, "output_tritower", "final_result.json")
    if not os.path.exists(json_path):
        print(f"  [INFO] final_result.json not found, using default RGCN AUC={DEFAULT_RGCN_AUC}")
        return DEFAULT_RGCN_AUC
    try:
        with open(json_path, 'r') as f:
            result = json.load(f)
        auc = float(result["tower_test"]["rgcn"]["auc"])
        print(f"  Loaded RGCN test AUC from final_result.json: {auc:.4f}")
        return auc
    except (KeyError, ValueError, TypeError):
        print(f"  [WARN] Could not parse RGCN AUC from final_result.json, using default={DEFAULT_RGCN_AUC}")
        return DEFAULT_RGCN_AUC


def load_center_features():
    train_structures, train_labels = config.load_structures_and_labels(
        os.path.join(config.DATA_DIR, "train_ss_41.fasta"), 3700, 37000
    )
    test_structures, test_labels = config.load_structures_and_labels(
        os.path.join(config.DATA_DIR, "test_ss_41.fasta"), 320, 320
    )
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    def extract_center(structures, desc=""):
        feats = []
        for i, (seq, ss) in enumerate(structures):
            f = build_structural_node_features(seq, ss)
            feats.append(f[CENTER_POS].numpy())
        return np.array(feats)

    print("  Extracting train center features...")
    X_train = extract_center(train_structures)
    print("  Extracting test center features...")
    X_test = extract_center(test_structures)

    return X_train, train_labels, X_test, test_labels


def train_surrogate(X_train, y_train, X_test, y_test):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, train_auc, test_auc


def compute_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    return shap_values


def plot_shap_bar(shap_values, X_test, out_dir):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(FEATURE_NAMES))
    ax.barh(y_pos, mean_abs_shap[sorted_idx], color="#1f77b4", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_shap_bar_importance.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_shap_bar_importance.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_shap_bar_importance.png/pdf")


def plot_shap_dependence(shap_values, X_test, out_dir):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs_shap)[::-1][:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, feat_idx in enumerate(top3_idx):
        ax = axes[i]
        ax.scatter(X_test[:, feat_idx], shap_values[:, feat_idx],
                   c=X_test[:, feat_idx], cmap="viridis", s=10, alpha=0.6)
        ax.set_xlabel(FEATURE_NAMES[feat_idx], fontsize=11)
        ax.set_ylabel("SHAP value", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_shap_dependence_top3.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_shap_dependence_top3.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_shap_dependence_top3.png/pdf")


def plot_shap_beeswarm(shap_values, X_test, out_dir):
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_NAMES,
                      show=False)
    fig = plt.gcf()
    fig.savefig(os.path.join(out_dir, "fig_shap_summary_beeswarm.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_shap_summary_beeswarm.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_shap_summary_beeswarm.png/pdf")


def main():
    print("=" * 70)
    print("  TriTower-m6Am Interpretability: SHAP on RGCN Structural Features")
    print("=" * 70)

    print("\n[1/5] Loading structural features for center node (pos 20)...")
    X_train, y_train, X_test, y_test = load_center_features()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Feature dim: {X_train.shape[1]}")

    rgcn_auc = load_rgcn_auc()

    print("\n[2/5] Training XGBoost surrogate model...")
    model, train_auc, test_auc = train_surrogate(X_train, y_train, X_test, y_test)
    print(f"  Surrogate train AUC: {train_auc:.4f}")
    print(f"  Surrogate test AUC:  {test_auc:.4f}")
    print(f"  RGCN single tower AUC: {rgcn_auc:.4f}")
    if test_auc < rgcn_auc:
        print(f"  [OK] Surrogate AUC < RGCN AUC ({test_auc:.4f} < {rgcn_auc:.4f})")
        print(f"       => Single-node features insufficient, graph structure is essential")
    else:
        print(f"  [WARNING] Surrogate AUC >= RGCN AUC")

    print("\n[3/5] Computing SHAP values...")
    shap_values = compute_shap(model, X_test)
    print(f"  SHAP values shape: {shap_values.shape}")

    print("\n[4/5] Generating figures...")
    plot_shap_bar(shap_values, X_test, OUT_DIR)
    plot_shap_dependence(shap_values, X_test, OUT_DIR)
    plot_shap_beeswarm(shap_values, X_test, OUT_DIR)

    print("\n[5/5] Saving results...")
    npz_path = os.path.join(OUT_DIR, "shap_results.npz")
    np.savez(
        npz_path,
        shap_values=shap_values,
        X_test=X_test,
        feature_names=np.array(FEATURE_NAMES),
        mean_abs_shap=np.abs(shap_values).mean(axis=0),
        surrogate_train_auc=train_auc,
        surrogate_test_auc=test_auc,
        rgcn_single_tower_auc=rgcn_auc,
    )
    print(f"  Saved: {npz_path}")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top5_features = [(FEATURE_NAMES[i], float(mean_abs_shap[i])) for i in sorted_idx[:5]]

    summary = {
        "surrogate_train_auc": float(train_auc),
        "surrogate_test_auc": float(test_auc),
        "rgcn_single_tower_auc": float(rgcn_auc),
        "auc_gap": float(rgcn_auc - test_auc),
        "interpretation": "Single-node features insufficient" if test_auc < rgcn_auc else "Surrogate matches RGCN",
        "top5_features": top5_features,
        "feature_importance_ranking": [(FEATURE_NAMES[i], float(mean_abs_shap[i])) for i in sorted_idx],
    }
    json_path = os.path.join(OUT_DIR, "shap_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 70)
    print("  Summary for paper:")
    print(f"  - Surrogate AUC: {test_auc:.4f} vs RGCN AUC: {rgcn_auc:.4f} (gap={rgcn_auc-test_auc:.4f})")
    print(f"  - Top-5 features: {top5_features}")
    print("=" * 70)


if __name__ == "__main__":
    main()
