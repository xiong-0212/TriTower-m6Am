import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from model import RNAFMTower, KmerBiLSTMTower, StructureRGCNTower
from utils.dataset import (RNAFMDataset, OnehotDataset, GraphDataset,
                           graph_collate_fn, prepare_tritower_dataset)
from utils.metrics import compute_metrics, find_optimal_mcc, threshold_scan, balanced_oof_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 41
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (self.decay * self.shadow[name] +
                                     (1 - self.decay) * param.data)

    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def predict_tower(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch[0], torch.Tensor):
                feat, label = batch
                feat = feat.to(DEVICE)
                with autocast('cuda'):
                    logits = model(feat)
            else:
                graph_data, label = batch
                graph_data = graph_data.to(DEVICE)
                with autocast('cuda'):
                    logits = model(graph_data)
            probs = torch.sigmoid(logits).squeeze(-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(label.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def train_tower(model, train_loader, val_loader, seed, epochs=150, lr=5e-4,
                weight_decay=5e-3, patience=50, ema_decay=0.990):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_pos = sum(1 for _, lb in train_loader.dataset if lb.item() == 1)
    n_neg = len(train_loader.dataset) - n_pos
    pos_weight = torch.tensor([min(n_neg / max(n_pos, 1), 3.0)],
                              dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs)

    ema = EMA(model, decay=ema_decay)
    scaler = GradScaler('cuda')
    best_val_mcc = -1.0
    no_improve = 0
    best_shadow = None

    val_probs_all, val_labels_all = predict_tower(model, val_loader)
    pos_mask = val_labels_all == 1
    n_val_pos = pos_mask.sum()
    neg_indices = np.where(~pos_mask)[0]
    rng = np.random.RandomState(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            if isinstance(batch[0], torch.Tensor):
                feat, label = batch
                feat = feat.to(DEVICE)
            else:
                feat, label = batch
                feat = feat.to(DEVICE)
            label = label.to(DEVICE)
            label_smooth = label * 0.92 + 0.04

            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(feat).squeeze(-1)
                loss = criterion(logits, label_smooth)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            total_loss += loss.item()

        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr * warmup_factor
        else:
            scheduler.step()

        ema.apply(model)
        val_probs, val_labels = predict_tower(model, val_loader)

        sampled_neg = rng.choice(neg_indices, size=min(n_val_pos, len(neg_indices)),
                                 replace=False)
        balanced_idx = np.concatenate([np.where(pos_mask)[0], sampled_neg])
        bal_probs = val_probs[balanced_idx]
        bal_labels = val_labels[balanced_idx]
        val_mcc, _ = find_optimal_mcc(bal_labels, bal_probs)

        ema.restore(model)

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            no_improve = 0
            best_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_shadow is not None:
        ema.shadow = best_shadow
    ema.apply(model)
    return model, ema


def train_all_towers(train_rnafm, train_onehot, train_labels, train_graphs,
                     test_rnafm, test_onehot, test_labels, test_graphs,
                     n_splits=5, seed=123):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = {'rnafm': np.zeros(len(train_labels), dtype=np.float32),
                 'onehot': np.zeros(len(train_labels), dtype=np.float32),
                 'rgcn': np.zeros(len(train_labels), dtype=np.float32)}
    test_preds = {'rnafm': np.zeros(len(test_labels), dtype=np.float32),
                  'onehot': np.zeros(len(test_labels), dtype=np.float32),
                  'rgcn': np.zeros(len(test_labels), dtype=np.float32)}

    ckpt_dir = os.path.join(OUTPUT_DIR, f"ckpt_seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_rnafm, train_labels)):
        fold_ckpt = os.path.join(ckpt_dir, f"fold{fold_idx}.npz")
        fold_pths_exist = all(
            os.path.exists(os.path.join(ckpt_dir, f"{t}_fold{fold_idx}.pth"))
            for t in ['rnafm', 'onehot', 'rgcn']
        )
        if os.path.exists(fold_ckpt) and fold_pths_exist:
            print(f"\n  Fold {fold_idx + 1}/{n_splits} (seed={seed}) -- SKIP (checkpoint found)")
            ckpt = np.load(fold_ckpt, allow_pickle=True)
            oof_preds['rnafm'][val_idx] = ckpt['oof_rnafm']
            oof_preds['onehot'][val_idx] = ckpt['oof_onehot']
            oof_preds['rgcn'][val_idx] = ckpt['oof_rgcn']
            test_preds['rnafm'] += ckpt['test_rnafm'] / n_splits
            test_preds['onehot'] += ckpt['test_onehot'] / n_splits
            test_preds['rgcn'] += ckpt['test_rgcn'] / n_splits
            continue

        print(f"\n  Fold {fold_idx + 1}/{n_splits} (seed={seed})")

        tr_rnafm = train_rnafm[train_idx]
        va_rnafm = train_rnafm[val_idx]
        tr_onehot = train_onehot[train_idx]
        va_onehot = train_onehot[val_idx]
        tr_labels = train_labels[train_idx]
        va_labels = train_labels[val_idx]
        tr_graphs = [train_graphs[i] for i in train_idx]
        va_graphs = [train_graphs[i] for i in val_idx]

        ds_rnafm_tr = RNAFMDataset(tr_rnafm, tr_labels)
        ds_rnafm_va = RNAFMDataset(va_rnafm, va_labels)
        ds_onehot_tr = OnehotDataset(tr_onehot, tr_labels)
        ds_onehot_va = OnehotDataset(va_onehot, va_labels)
        ds_graph_tr = GraphDataset(tr_graphs, tr_labels)
        ds_graph_va = GraphDataset(va_graphs, va_labels)

        dl_kw = dict(batch_size=64, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=4)
        dl_rnafm_tr = DataLoader(ds_rnafm_tr, shuffle=True, **dl_kw)
        dl_rnafm_va = DataLoader(ds_rnafm_va, shuffle=False, **dl_kw)
        dl_onehot_tr = DataLoader(ds_onehot_tr, shuffle=True, **dl_kw)
        dl_onehot_va = DataLoader(ds_onehot_va, shuffle=False, **dl_kw)
        dl_graph_tr = DataLoader(ds_graph_tr, shuffle=True, collate_fn=graph_collate_fn,
                                 **dl_kw)
        dl_graph_va = DataLoader(ds_graph_va, shuffle=False, collate_fn=graph_collate_fn,
                                 **dl_kw)

        ds_rnafm_te = RNAFMDataset(test_rnafm, test_labels)
        ds_onehot_te = OnehotDataset(test_onehot, test_labels)
        ds_graph_te = GraphDataset(test_graphs, test_labels)
        dl_rnafm_te = DataLoader(ds_rnafm_te, shuffle=False, **dl_kw)
        dl_onehot_te = DataLoader(ds_onehot_te, shuffle=False, **dl_kw)
        dl_graph_te = DataLoader(ds_graph_te, shuffle=False, collate_fn=graph_collate_fn,
                                 **dl_kw)

        torch.manual_seed(seed + fold_idx)
        model_rnafm = RNAFMTower().to(DEVICE)
        best_rnafm, _ = train_tower(model_rnafm, dl_rnafm_tr, dl_rnafm_va,
                                     seed + fold_idx)
        oof_rnafm, _ = predict_tower(best_rnafm, dl_rnafm_va)
        test_rnafm_p, _ = predict_tower(best_rnafm, dl_rnafm_te)
        oof_preds['rnafm'][val_idx] = oof_rnafm
        test_preds['rnafm'] += test_rnafm_p / n_splits

        torch.manual_seed(seed + fold_idx + 100)
        model_onehot = KmerBiLSTMTower().to(DEVICE)
        best_onehot, _ = train_tower(model_onehot, dl_onehot_tr, dl_onehot_va,
                                      seed + fold_idx + 100)
        oof_onehot, _ = predict_tower(best_onehot, dl_onehot_va)
        test_onehot_p, _ = predict_tower(best_onehot, dl_onehot_te)
        oof_preds['onehot'][val_idx] = oof_onehot
        test_preds['onehot'] += test_onehot_p / n_splits

        torch.manual_seed(seed + fold_idx + 200)
        model_rgcn = StructureRGCNTower().to(DEVICE)
        best_rgcn, _ = train_tower(model_rgcn, dl_graph_tr, dl_graph_va,
                                    seed + fold_idx + 200)
        oof_rgcn, _ = predict_tower(best_rgcn, dl_graph_va)
        test_rgcn_p, _ = predict_tower(best_rgcn, dl_graph_te)
        oof_preds['rgcn'][val_idx] = oof_rgcn
        test_preds['rgcn'] += test_rgcn_p / n_splits

        torch.save(best_rnafm.state_dict(), os.path.join(ckpt_dir, f"rnafm_fold{fold_idx}.pth"))
        torch.save(best_onehot.state_dict(), os.path.join(ckpt_dir, f"onehot_fold{fold_idx}.pth"))
        torch.save(best_rgcn.state_dict(), os.path.join(ckpt_dir, f"rgcn_fold{fold_idx}.pth"))

        np.savez(fold_ckpt,
                 oof_rnafm=oof_preds['rnafm'][val_idx],
                 oof_onehot=oof_preds['onehot'][val_idx],
                 oof_rgcn=oof_preds['rgcn'][val_idx],
                 test_rnafm=test_rnafm_p,
                 test_onehot=test_onehot_p,
                 test_rgcn=test_rgcn_p)
        print(f"    Checkpoint saved: fold{fold_idx}")

    return oof_preds, test_preds


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def main():
    SEED = 123
    N_SPLITS = 5

    print("=" * 70)
    print("  TriTower-m6Am (Seed=123)")
    print("  Hyperparameters: lr=5e-4, wd=5e-3, dropout=0.4, ema=0.990")
    print("  RNA-FM Tower + One-hot BiLSTM Tower + RGCN Tower")
    print("  AUC Weighted Ensemble + Balanced-OOF Threshold")
    print("=" * 70)

    print("Loading TriTower dataset...")
    (train_rnafm, train_onehot, train_labels, train_graphs,
     test_rnafm, test_onehot, test_labels, test_graphs) = prepare_tritower_dataset()
    print(f"  Train: rnafm={train_rnafm.shape}, labels={len(train_labels)}")
    print(f"  Test:  rnafm={test_rnafm.shape}, labels={len(test_labels)}")

    print(f"\n{'#' * 70}")
    print(f"  SEED = {SEED}")
    print(f"{'#' * 70}")

    oof_preds, test_preds = train_all_towers(
        train_rnafm, train_onehot, train_labels, train_graphs,
        test_rnafm, test_onehot, test_labels, test_graphs,
        n_splits=N_SPLITS, seed=SEED
    )

    print(f"\n--- Per-Tower OOF Metrics (seed={SEED}) ---")
    for name in ['rnafm', 'onehot', 'rgcn']:
        m = compute_metrics(train_labels, oof_preds[name])
        print(f"  {name:8s}: AUC={m['auc']:.4f}  MCC={m['mcc']:.4f}  "
              f"SN={m['sn']:.4f}  SP={m['sp']:.4f}")

    print(f"\n--- Per-Tower Test Metrics (seed={SEED}) ---")
    for name in ['rnafm', 'onehot', 'rgcn']:
        thr = balanced_oof_threshold(train_labels, oof_preds[name])
        m = compute_metrics(test_labels, test_preds[name], threshold=thr)
        print(f"  {name:8s}: AUC={m['auc']:.4f}  MCC={m['mcc']:.4f}  "
              f"SN={m['sn']:.4f}  SP={m['sp']:.4f}  (thr={thr:.2f})")

    print(f"\n--- AUC Weighted Ensemble ---")
    oof_arr = np.column_stack([oof_preds[n] for n in ['rnafm', 'onehot', 'rgcn']])
    test_arr = np.column_stack([test_preds[n] for n in ['rnafm', 'onehot', 'rgcn']])
    oof_aucs = [roc_auc_score(train_labels, oof_arr[:, i]) for i in range(3)]
    auc_w = np.array(oof_aucs)
    auc_w = auc_w / auc_w.sum()
    print(f"  Weights: rnafm={auc_w[0]:.4f} onehot={auc_w[1]:.4f} rgcn={auc_w[2]:.4f}")
    oof_ensemble = (oof_arr * auc_w).sum(axis=1)
    final_probs = (test_arr * auc_w).sum(axis=1)

    # Threshold is selected once on the class-balanced OOF subset and held
    # fixed for the test set (never tuned on evaluation data).
    final_thr = balanced_oof_threshold(train_labels, oof_ensemble)
    final_metrics = compute_metrics(test_labels, final_probs, threshold=final_thr)
    print(f"\n  Balanced-OOF Threshold: {final_thr:.2f}")
    print(f"  AUC={final_metrics['auc']:.4f}  MCC={final_metrics['mcc']:.4f}  "
          f"ACC={final_metrics['acc']:.4f}  F1={final_metrics['f1']:.4f}  "
          f"SN={final_metrics['sn']:.4f}  SP={final_metrics['sp']:.4f}")

    print(f"\n--- Threshold Scan (SN/SP/MCC Trade-off) ---")
    threshold_scan(test_labels, final_probs)

    print(f"\n{'=' * 70}")
    print(f"  Comparison vs DTC-m6Am (AUC=0.765, MCC=0.411)")
    print(f"{'=' * 70}")
    print(f"  TriTower-m6Am: AUC={final_metrics['auc']:.4f} "
          f"(+{final_metrics['auc'] - 0.765:.4f})  "
          f"MCC={final_metrics['mcc']:.4f} "
          f"(+{final_metrics['mcc'] - 0.411:.4f})")

    final_result_file = os.path.join(OUTPUT_DIR, "final_result.json")
    with open(final_result_file, 'w') as f:
        json.dump(make_serializable({
            'seed': SEED,
            'hyperparameters': {
                'lr': 5e-4,
                'weight_decay': 5e-3,
                'dropout': 0.4,
                'ema_decay': 0.990,
            },
            'tower_oof': {n: compute_metrics(train_labels, oof_preds[n])
                         for n in ['rnafm', 'onehot', 'rgcn']},
            'tower_test': {n: compute_metrics(
                                test_labels, test_preds[n],
                                threshold=balanced_oof_threshold(train_labels, oof_preds[n]))
                          for n in ['rnafm', 'onehot', 'rgcn']},
            'ensemble_weights': {n: float(w)
                               for n, w in zip(['rnafm', 'onehot', 'rgcn'], auc_w)},
            'final': final_metrics,
            'vs_dtc': {'auc_diff': final_metrics['auc'] - 0.765,
                      'mcc_diff': final_metrics['mcc'] - 0.411},
        }), f, indent=2)
    print(f"\n  Final result saved: {final_result_file}")


if __name__ == '__main__':
    main()