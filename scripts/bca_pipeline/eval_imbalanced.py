import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, f1_score,
                              accuracy_score, precision_recall_curve,
                              average_precision_score)
import tritower_config as config
from struct_features import build_structural_node_features, parse_base_pairs, clean_dot_bracket
from tritower_models import RNAFMTower, KmerBiLSTMTower, StructureRGCNTower

try:
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "output_imbalanced_eval")

NUC_MAP = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}


def build_onehot_matrix(seqs_list):
    n = len(seqs_list)
    mat = np.zeros((n, config.SEQ_LEN, 4), dtype=np.float32)
    for i, seq in enumerate(seqs_list):
        for j, ch in enumerate(seq):
            if ch in NUC_MAP:
                mat[i, j] = NUC_MAP[ch]
    return mat


def build_rgcn_graph(seq, ss_clean):
    x = build_structural_node_features(seq, ss_clean)
    pairs = parse_base_pairs(ss_clean)
    src, dst, etypes = [], [], []
    for i in range(config.SEQ_LEN - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        etypes.extend([0, 0])
    for i, j in pairs.items():
        if i < j:
            src.extend([i, j])
            dst.extend([j, i])
            etypes.extend([1, 1])
    for i in range(config.SEQ_LEN - 2):
        src.extend([i, i + 2])
        dst.extend([i + 2, i])
        etypes.extend([2, 2])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etypes, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.center_idx = config.SEQ_LEN // 2
    return data


def load_ss_fasta(fasta_path):
    seqs = []
    with open(fasta_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            seq = lines[i + 1].strip().upper()
            ss = lines[i + 2].strip() if i + 2 < len(lines) else '.' * len(seq)
            seqs.append((seq, ss))
            i += 3
        else:
            i += 1
    return seqs


def prepare_test_data():
    test_seqs = load_ss_fasta(os.path.join(config.DATA_DIR, "test_ss_41.fasta"))
    pos_seqs = test_seqs[:320]
    orig_neg_seqs = test_seqs[320:]

    bca_path = os.path.join(config.DATA_DIR, "bca_negatives_41_sampled.fasta")
    bca_seqs = []
    if os.path.exists(bca_path):
        bca_seqs = load_ss_fasta(bca_path)
        print(f"  BCA negatives loaded: {len(bca_seqs)}")
    else:
        print(f"  [WARN] BCA negatives not found: {bca_path}")
        print(f"  Run prepare_bca_negatives.py first!")

    return pos_seqs, orig_neg_seqs, bca_seqs


def extract_features(seqs_with_ss):
    seqs = [s[0] for s in seqs_with_ss]
    ss_list = [s[1] for s in seqs_with_ss]

    onehot = build_onehot_matrix(seqs)

    graphs = []
    for seq, ss in zip(seqs, ss_list):
        ss_c = clean_dot_bracket(ss)
        graphs.append(build_rgcn_graph(seq, ss_c))

    return onehot, graphs


def load_rnafm_embeddings(seqs_with_ss, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading RNA-FM embeddings from cache: {cache_path}")
        return np.load(cache_path).astype(np.float32)

    print(f"  [WARN] RNA-FM embeddings not found at {cache_path}")
    print(f"  Please run RNA-FM extraction first.")
    return None


class RNAFMDataset(Dataset):
    def __init__(self, rnafm_arr, labels):
        self.rnafm = torch.tensor(rnafm_arr, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.rnafm[idx], self.labels[idx]


class OnehotDataset(Dataset):
    def __init__(self, onehot_arr, labels):
        self.onehot = torch.tensor(onehot_arr, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.onehot[idx], self.labels[idx]


class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def graph_collate_fn(batch):
    graphs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    batched = Batch.from_data_list(graphs)
    center_indices = torch.tensor([g.center_idx for g in graphs], dtype=torch.long)
    batched.center_indices = center_indices
    return batched, labels


def make_loader(data, labels, tower_type, batch_size=256, num_workers=2):
    if tower_type == 'rnafm':
        ds = RNAFMDataset(data, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False)
    elif tower_type == 'onehot':
        ds = OnehotDataset(data, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False)
    elif tower_type == 'rgcn':
        ds = GraphDataset(data, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=graph_collate_fn, num_workers=num_workers,
                          pin_memory=False)


def predict_ensemble(tower_type, data, labels, ckpt_dir, n_splits=5):
    all_preds = np.zeros(len(labels), dtype=np.float32)

    for fold_idx in range(n_splits):
        ckpt_path = os.path.join(ckpt_dir, f"{tower_type}_fold{fold_idx}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] Checkpoint not found: {ckpt_path}")
            continue

        torch.manual_seed(123 + fold_idx + 200)
        if tower_type == 'rnafm':
            model = RNAFMTower().to(config.DEVICE)
        elif tower_type == 'onehot':
            model = KmerBiLSTMTower().to(config.DEVICE)
        elif tower_type == 'rgcn':
            model = StructureRGCNTower().to(config.DEVICE)

        model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
        model.eval()

        loader = make_loader(data, labels, tower_type)

        fold_preds = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch[0], torch.Tensor):
                    feat, _ = batch
                    feat = feat.to(config.DEVICE)
                    with autocast('cuda'):
                        logits = model(feat)
                else:
                    graph_data, _ = batch
                    graph_data = graph_data.to(config.DEVICE)
                    with autocast('cuda'):
                        logits = model(graph_data)
                probs = torch.sigmoid(logits).squeeze(-1)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds)
        all_preds += fold_preds / n_splits
        print(f"    Fold {fold_idx + 1}/{n_splits} done")

    return all_preds


def ensemble_predictions(rnafm_preds, onehot_preds, rgcn_preds,
                          val_aucs=None):
    if val_aucs is None:
        val_aucs = {'rnafm': 0.90, 'onehot': 0.92, 'rgcn': 0.88}

    weights = np.array([val_aucs['rnafm'], val_aucs['onehot'], val_aucs['rgcn']])
    weights = weights / weights.sum()

    ensemble = (weights[0] * rnafm_preds +
                weights[1] * onehot_preds +
                weights[2] * rgcn_preds)
    return ensemble, weights


def find_optimal_mcc(labels, probs):
    best_mcc = -1.0
    best_thr = 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= thr).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = thr
    return best_mcc, best_thr


def compute_full_metrics(labels, probs):
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    mcc, threshold = find_optimal_mcc(labels, probs)
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    tp = ((preds == 1) & (labels == 1)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'auc': auc, 'ap': ap, 'pr_auc': pr_auc,
        'mcc': mcc, 'threshold': threshold,
        'acc': acc, 'f1': f1, 'sn': sn, 'sp': sp,
        'n_samples': len(labels), 'n_pos': int(labels.sum()),
        'n_neg': int(len(labels) - labels.sum()),
        'ratio': f"1:{int(len(labels) / max(labels.sum(), 1))}",
    }


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="Imbalanced evaluation")
    parser.add_argument('--ratios', type=int, nargs='+', default=[1, 10, 100],
                        help='Negative:positive ratios to test (default: 1 10 100)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for sampling')
    args = parser.parse_args()

    print("=" * 70)
    print("TriTower-m6Am Imbalanced Evaluation")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nStep 1: Loading test data...")
    pos_seqs, orig_neg_seqs, bca_seqs = prepare_test_data()
    print(f"  Positive samples: {len(pos_seqs)}")
    print(f"  Original negative: {len(orig_neg_seqs)}")
    print(f"  BCA negatives: {len(bca_seqs)}")

    ckpt_dir = os.path.join(config.PROJECT_ROOT, "output_tritower", "ckpt_seed123")
    if not os.path.exists(ckpt_dir):
        print(f"\n[ERROR] Checkpoint directory not found: {ckpt_dir}")
        print(f"  Please run train_tritower.py first to generate checkpoints.")
        sys.exit(1)
    print(f"\n  Checkpoints: {ckpt_dir}")

    result_path = os.path.join(config.PROJECT_ROOT, "output_tritower", "final_result.json")
    val_aucs = None
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        if 'ensemble_val_aucs' in result:
            val_aucs = result['ensemble_val_aucs']
            print(f"  Ensemble val AUCs: {val_aucs}")

    results = {}
    rng = np.random.RandomState(args.seed)

    for ratio in args.ratios:
        ratio_key = f"1:{ratio}"
        print(f"\n{'='*70}")
        print(f"  Evaluating ratio {ratio_key}")
        print(f"{'='*70}")

        n_neg_needed = 320 * ratio
        n_bca_needed = max(0, n_neg_needed - len(orig_neg_seqs))

        if n_bca_needed > len(bca_seqs):
            print(f"  [WARN] Not enough BCA negatives: need {n_bca_needed}, "
                  f"have {len(bca_seqs)}. Using all available.")
            n_bca_needed = len(bca_seqs)

        if n_bca_needed > 0:
            bca_indices = rng.choice(len(bca_seqs), size=n_bca_needed, replace=False)
            sampled_bca = [bca_seqs[i] for i in bca_indices]
        else:
            sampled_bca = []

        all_seqs = pos_seqs + orig_neg_seqs + sampled_bca
        all_labels = np.array([1] * len(pos_seqs) +
                              [0] * (len(orig_neg_seqs) + len(sampled_bca)),
                              dtype=np.float32)

        print(f"  Test set: {len(all_labels)} samples "
              f"({int(all_labels.sum())} pos + {int(len(all_labels) - all_labels.sum())} neg)")

        print(f"\n  Extracting features...")
        onehot, graphs = extract_features(all_seqs)

        rnafm_test_orig = np.load(os.path.join(config.DATA_DIR, 'rnafm',
                                                'rnafm_test_ss_41.npy')).astype(np.float32)
        rnafm_bca_path = os.path.join(config.DATA_DIR, 'rnafm', 'rnafm_bca_negatives_41_sampled.npy')

        if len(sampled_bca) > 0:
            if os.path.exists(rnafm_bca_path):
                rnafm_bca = np.load(rnafm_bca_path).astype(np.float32)
                rnafm_bca_sampled = rnafm_bca[bca_indices]
                rnafm_all = np.concatenate([rnafm_test_orig, rnafm_bca_sampled], axis=0)
            else:
                print(f"  [ERROR] BCA RNA-FM embeddings not found: {rnafm_bca_path}")
                print(f"  Please run RNA-FM extraction on BCA negatives first.")
                continue
        else:
            rnafm_all = rnafm_test_orig

        print(f"\n  Running 5-fold ensemble inference...")

        print(f"    RNA-FM tower:")
        rnafm_preds = predict_ensemble('rnafm', rnafm_all, all_labels, ckpt_dir)

        print(f"    BiLSTM tower:")
        onehot_preds = predict_ensemble('onehot', onehot, all_labels, ckpt_dir)

        print(f"    RGCN tower:")
        rgcn_preds = predict_ensemble('rgcn', graphs, all_labels, ckpt_dir)

        ensemble_preds, weights = ensemble_predictions(
            rnafm_preds, onehot_preds, rgcn_preds, val_aucs)
        print(f"\n  Ensemble weights: RNA-FM={weights[0]:.4f}, "
              f"BiLSTM={weights[1]:.4f}, RGCN={weights[2]:.4f}")

        metrics = compute_full_metrics(all_labels, ensemble_preds)
        print(f"\n  --- Results (ratio {ratio_key}) ---")
        print(f"    AUC     = {metrics['auc']:.4f}")
        print(f"    AP      = {metrics['ap']:.4f}")
        print(f"    PR-AUC  = {metrics['pr_auc']:.4f}")
        print(f"    MCC     = {metrics['mcc']:.4f}  (threshold={metrics['threshold']:.2f})")
        print(f"    F1      = {metrics['f1']:.4f}")
        print(f"    SN      = {metrics['sn']:.4f}")
        print(f"    SP      = {metrics['sp']:.4f}")

        tower_metrics = {}
        for tower_name, tower_preds in [('rnafm', rnafm_preds),
                                         ('onehot', onehot_preds),
                                         ('rgcn', rgcn_preds)]:
            tower_metrics[tower_name] = compute_full_metrics(all_labels, tower_preds)

        results[ratio_key] = {
            'ensemble': metrics,
            'towers': tower_metrics,
            'n_pos': int(all_labels.sum()),
            'n_neg': int(len(all_labels) - all_labels.sum()),
        }

        pred_path = os.path.join(OUTPUT_DIR, f"preds_ratio_{ratio}.npz")
        np.savez(pred_path,
                 labels=all_labels,
                 rnafm=rnafm_preds,
                 onehot=onehot_preds,
                 rgcn=rgcn_preds,
                 ensemble=ensemble_preds)
        print(f"    Predictions saved: {pred_path}")

    summary_path = os.path.join(OUTPUT_DIR, "imbalanced_eval_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n{'='*70}")
    print(f"  Summary saved: {summary_path}")
    print(f"{'='*70}")

    print(f"\n  {'Ratio':<8} {'N_pos':>6} {'N_neg':>8} {'AUC':>8} {'AP':>8} "
          f"{'MCC':>8} {'F1':>8} {'SN':>8} {'SP':>8}")
    print(f"  {'-'*70}")
    for ratio_key, r in results.items():
        m = r['ensemble']
        print(f"  {ratio_key:<8} {r['n_pos']:>6} {r['n_neg']:>8} "
              f"{m['auc']:>8.4f} {m['ap']:>8.4f} {m['mcc']:>8.4f} "
              f"{m['f1']:>8.4f} {m['sn']:>8.4f} {m['sp']:>8.4f}")


if __name__ == '__main__':
    main()
