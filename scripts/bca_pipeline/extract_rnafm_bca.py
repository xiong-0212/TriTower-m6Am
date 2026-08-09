#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RNAFM_DIR = os.path.join(SCRIPT_DIR, "RNA-FM-main")
sys.path.insert(0, RNAFM_DIR)

import fm

MODEL_PATH = os.path.join(RNAFM_DIR, "pretrained", "RNA-FM_pretrained.pth")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "rnafm")

BCA_FASTA = os.path.join(DATA_DIR, "bca_negatives_41_sampled.fasta")
OUTPUT_NPY = os.path.join(OUTPUT_DIR, "rnafm_bca_negatives_41_sampled.npy")

BATCH_SIZE = 8
SEQ_LEN = 41

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


print(f"Loading RNA-FM model from: {MODEL_PATH}")
model, alphabet = fm.pretrained.rna_fm_t12(model_location=MODEL_PATH)
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()


def load_sequences_from_ss(ss_file):
    seqs = []
    with open(ss_file) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            seq = lines[i + 1].strip().upper()
            seqs.append(seq)
            i += 3
        else:
            i += 1
    return seqs


def extract_embeddings(seqs):
    all_emb = []
    for start in tqdm(range(0, len(seqs), BATCH_SIZE), desc="Extracting embeddings"):
        batch_seqs = seqs[start:start + BATCH_SIZE]
        data = [(f"s{i}", s) for i, s in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])
            token_emb = results["representations"][12]
        emb = token_emb[:, 1:-1, :].cpu().numpy()
        all_emb.append(emb)
    return np.concatenate(all_emb, axis=0).astype(np.float32)


def main():
    print("=" * 60)
    print("RNA-FM Embedding Extraction for BCA Negatives")
    print("=" * 60)

    if not os.path.exists(BCA_FASTA):
        print(f"\n[ERROR] BCA FASTA not found: {BCA_FASTA}")
        print("Please run fix_bca_rnafold.py first to generate correct file.")
        sys.exit(1)

    print(f"\nLoading sequences from: {BCA_FASTA}")
    seqs = load_sequences_from_ss(BCA_FASTA)
    print(f"  Loaded {len(seqs)} sequences")

    seq_lens = [len(s) for s in seqs]
    print(f"  Sequence length range: {min(seq_lens)} - {max(seq_lens)}")
    if min(seq_lens) != 41 or max(seq_lens) != 41:
        print(f"  [ERROR] Expected all sequences to be 41 nt, got {min(seq_lens)}-{max(seq_lens)}")
        print(f"  Please run fix_bca_rnafold.py first to fix the file.")
        sys.exit(1)

    print(f"\nExtracting RNA-FM embeddings (batch_size={BATCH_SIZE})...")
    embeddings = extract_embeddings(seqs)
    print(f"  Embeddings shape: {embeddings.shape}")

    print(f"\nSaving to: {OUTPUT_NPY}")
    np.save(OUTPUT_NPY, embeddings)
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"\nDone! Next step: run eval_imbalanced.py")


if __name__ == "__main__":
    main()
