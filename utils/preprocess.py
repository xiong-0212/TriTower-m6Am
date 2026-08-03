import torch
import numpy as np
from torch_geometric.data import Data


def parse_base_pairs(ss):
    pairs = {}
    stack = []
    for i, c in enumerate(ss):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                pairs[j] = i
                pairs[i] = j
    return pairs


def clean_dot_bracket(ss):
    stack = []
    matched = [False] * len(ss)
    for i, c in enumerate(ss):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                matched[j] = True
                matched[i] = True
    return ''.join('(' if matched[i] and ss[i] == '(' else
                   ')' if matched[i] and ss[i] == ')' else
                   '.' for i in range(len(ss)))


def annotate_loop_type(ss, pairs):
    L = len(ss)
    loop_types = [5] * L
    pair_list = sorted([(i, pairs[i]) for i in pairs if i < pairs[i]])

    for i, j in pair_list:
        loop_types[i] = 0
        loop_types[j] = 0

    for i_open, j_close in pair_list:
        direct_children = []
        for a, b in pair_list:
            if a > i_open and b < j_close:
                is_direct = True
                for c, d in pair_list:
                    if (c, d) != (a, b) and c > i_open and d < j_close and c < a and d > b:
                        is_direct = False
                        break
                if is_direct:
                    direct_children.append((a, b))

        child_ranges = set()
        for a, b in direct_children:
            for k in range(a, b + 1):
                child_ranges.add(k)

        unpaired = [k for k in range(i_open + 1, j_close)
                    if ss[k] == '.' and k not in child_ranges]
        if not unpaired:
            continue

        n_children = len(direct_children)
        if n_children == 0:
            for p in unpaired:
                loop_types[p] = 1
        elif n_children == 1:
            ca, cb = direct_children[0]
            left = any(p < ca for p in unpaired)
            right = any(p > cb for p in unpaired)
            for p in unpaired:
                loop_types[p] = 2 if (left and right) else 3
        else:
            for p in unpaired:
                loop_types[p] = 4

    return loop_types


def compute_pairing_density(ss, window=11):
    L = len(ss)
    paired = [1 if c in '()' else 0 for c in ss]
    densities = []
    half_w = window // 2
    for i in range(L):
        start = max(0, i - half_w)
        end = min(L, i + half_w + 1)
        region = paired[start:end]
        densities.append(sum(region) / len(region) if region else 0.0)
    return densities


def compute_stacking_depth(ss, pairs):
    L = len(ss)
    stacking = [0.0] * L
    pair_list = sorted([(i, pairs[i]) for i in pairs if i < pairs[i]])
    for i, j in pair_list:
        depth = 1
        ni, nj = i + 1, j - 1
        while ni < nj and ni in pairs and pairs[ni] == nj:
            depth += 1
            ni += 1
            nj -= 1
        for k in range(i, ni):
            stacking[k] = max(stacking[k], depth / 10.0)
        for k in range(j, nj, -1):
            stacking[k] = max(stacking[k], depth / 10.0)
    return stacking


def compute_gc_content(sequence, window=11):
    L = len(sequence)
    gc_content = []
    half_w = window // 2
    for i in range(L):
        start = max(0, i - half_w)
        end = min(L, i + half_w + 1)
        region = sequence[start:end]
        gc = sum(1 for c in region if c in 'GC') / len(region) if region else 0
        gc_content.append(gc)
    return gc_content


def build_structural_node_features(sequence, ss):
    L = len(sequence)
    pairs = parse_base_pairs(ss)
    loop_types = annotate_loop_type(ss, pairs)

    loop_lengths = [0.0] * L
    i = 0
    while i < L:
        if ss[i] == '.':
            j = i
            while j < L and ss[j] == '.':
                j += 1
            length = j - i
            norm_len = float(length) / float(L) if L > 0 else 0.0
            for k in range(i, j):
                loop_lengths[k] = norm_len
            i = j
        else:
            i += 1

    max_dist = float(L)
    pair_dists = []
    for i in range(L):
        if i in pairs:
            pair_dists.append(abs(pairs[i] - i) / max_dist)
        else:
            pair_dists.append(0.0)

    pair_densities = compute_pairing_density(ss, window=11)
    stacking = compute_stacking_depth(ss, pairs)
    gc_content = compute_gc_content(sequence, window=11)

    nuc_map = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0],
               'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    ss_map = {'(': [1, 0, 0], ')': [0, 1, 0], '.': [0, 0, 1]}
    loop_map = {
        0: [1, 0, 0, 0, 0, 0], 1: [0, 1, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0], 3: [0, 0, 0, 1, 0, 0],
        4: [0, 0, 0, 0, 1, 0], 5: [0, 0, 0, 0, 0, 1]
    }

    features = []
    for i in range(L):
        nuc_feat = nuc_map.get(sequence[i], [0, 0, 0, 0])
        ss_feat = ss_map.get(ss[i], [0, 0, 0])
        loop_feat = loop_map.get(loop_types[i], [0, 0, 0, 0, 0, 1])
        pos_sin = np.sin(2 * np.pi * i / L)
        pos_cos = np.cos(2 * np.pi * i / L)
        is_paired = 1.0 if i in pairs else 0.0

        feat = (nuc_feat + ss_feat + loop_feat +
                [pair_dists[i], pair_densities[i], stacking[i], loop_lengths[i],
                 pos_sin, pos_cos, is_paired, gc_content[i]])
        features.append(feat)

    return torch.tensor(features, dtype=torch.float32)


def build_rgcn_graph(sequence, ss, seq_len=41):
    x = build_structural_node_features(sequence, ss)
    pairs = parse_base_pairs(ss)
    src, dst, etypes = [], [], []

    for i in range(seq_len - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        etypes.extend([0, 0])

    for i, j in pairs.items():
        if i < j:
            src.extend([i, j])
            dst.extend([j, i])
            etypes.extend([1, 1])

    for i in range(seq_len - 2):
        src.extend([i, i + 2])
        dst.extend([i + 2, i])
        etypes.extend([2, 2])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etypes, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.center_idx = seq_len // 2
    return data