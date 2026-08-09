import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from utils.preprocess import build_rgcn_graph, clean_dot_bracket

NUC_MAP = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0],
           'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}


def build_onehot_matrix(seqs_list, seq_len=41):
    n = len(seqs_list)
    mat = np.zeros((n, seq_len, 4), dtype=np.float32)
    for i, seq in enumerate(seqs_list):
        for j, ch in enumerate(seq):
            if ch in NUC_MAP:
                mat[i, j] = NUC_MAP[ch]
    return mat


def load_structures_and_labels(ss_file, n_pos, n_neg):
    structures = []
    with open(ss_file) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            seq = lines[i + 1].strip().upper().replace('T', 'U')
            ss = lines[i + 2].strip()
            structures.append((seq, ss))
            i += 3
        else:
            i += 1
    assert len(structures) == n_pos + n_neg, \
        f"Expected {n_pos + n_neg} sequences, got {len(structures)}"
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32)
    return structures, labels


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


def prepare_tritower_dataset(data_dir="data", seq_len=41, edge_types=(0, 1, 2)):
    train_structures, train_labels = load_structures_and_labels(
        os.path.join(data_dir, "train_ss_41.fasta"), 3700, 37000)
    test_structures, test_labels = load_structures_and_labels(
        os.path.join(data_dir, "test_ss_41.fasta"), 320, 320)

    train_labels = np.array(train_labels, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)

    train_seqs = [s[0] for s in train_structures]
    train_ss = [s[1] for s in train_structures]
    test_seqs = [s[0] for s in test_structures]
    test_ss = [s[1] for s in test_structures]

    train_rnafm = np.load(os.path.join(data_dir, "rnafm", "rnafm_train_ss_41.npy")).astype(np.float32)
    test_rnafm = np.load(os.path.join(data_dir, "rnafm", "rnafm_test_ss_41.npy")).astype(np.float32)

    train_onehot = build_onehot_matrix(train_seqs, seq_len)
    test_onehot = build_onehot_matrix(test_seqs, seq_len)

    train_graphs = []
    for seq, ss in zip(train_seqs, train_ss):
        ss_c = clean_dot_bracket(ss)
        train_graphs.append(build_rgcn_graph(seq, ss_c, seq_len, edge_types))

    test_graphs = []
    for seq, ss in zip(test_seqs, test_ss):
        ss_c = clean_dot_bracket(ss)
        test_graphs.append(build_rgcn_graph(seq, ss_c, seq_len, edge_types))

    return (train_rnafm, train_onehot, train_labels, train_graphs,
            test_rnafm, test_onehot, test_labels, test_graphs)