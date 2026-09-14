import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score


def find_optimal_mcc(labels, probs):
    best_mcc = -1.0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= thr).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = thr
    return best_mcc, best_thr


def balanced_oof_threshold(oof_labels, oof_probs, seed=2026):
    """Select a decision threshold on a class-balanced out-of-fold subset.

    The test set is 1:1, so a class-balanced subset of the OOF predictions is
    used to pick the threshold by MCC; the value is then held fixed and applied
    to the test set, never tuned on evaluation data.
    """
    rng = np.random.RandomState(seed)
    pos = np.where(oof_labels == 1)[0]
    neg = np.where(oof_labels == 0)[0]
    n = min(len(pos), len(neg))
    sub = np.concatenate([rng.choice(pos, n, replace=False),
                          rng.choice(neg, n, replace=False)])
    _, thr = find_optimal_mcc(oof_labels[sub], oof_probs[sub])
    return thr


def compute_metrics(labels, probs, threshold=None):
    auc = roc_auc_score(labels, probs)
    if threshold is None:
        mcc, threshold = find_optimal_mcc(labels, probs)
    else:
        mcc = matthews_corrcoef(labels, (probs >= threshold).astype(int))

    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    tp = ((preds == 1) & (labels == 1)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {'auc': auc, 'mcc': mcc, 'acc': acc, 'f1': f1,
            'sn': sn, 'sp': sp, 'thresh': threshold}


def threshold_scan(labels, probs):
    print(f"\n  {'Threshold':>10} {'SN':>7} {'SP':>7} {'MCC':>7} {'F1':>7} {'ACC':>7}")
    print(f"  {'-' * 45}")
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        m = compute_metrics(labels, probs, threshold=thr)
        marker = " <-- SN>=0.9" if m['sn'] >= 0.9 else ""
        print(f"  {thr:>10.2f} {m['sn']:>7.4f} {m['sp']:>7.4f} "
              f"{m['mcc']:>7.4f} {m['f1']:>7.4f} {m['acc']:>7.4f}{marker}")