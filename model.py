import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, global_mean_pool


class BellPooling(nn.Module):
    def __init__(self, seq_len=41):
        super().__init__()
        self.seq_len = seq_len
        self.center = nn.Parameter(torch.tensor((seq_len - 1) / 2.0))
        self.b = nn.Parameter(torch.tensor(5.0))
        self.c = nn.Parameter(torch.tensor(2.0))
        self.d = nn.Parameter(torch.tensor(1.0))
        x = torch.arange(seq_len, dtype=torch.float32)
        with torch.no_grad():
            bell_w = self.d / (1.0 + torch.abs((x - self.center) / self.b).pow(self.c))
            bell_w = bell_w / bell_w.sum()
            self.logits = nn.Parameter(torch.log(bell_w + 1e-8))

    def forward(self):
        return torch.softmax(self.logits, dim=0)


class RNAFMTower(nn.Module):
    def __init__(self, seq_len=41, input_dim=640, proj_dim=96, cnn_channels=48, dropout=0.4):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, proj_dim)
        self.act_proj = nn.GELU()
        self.cnn_k3 = nn.Sequential(
            nn.Conv1d(proj_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        self.cnn_k5 = nn.Sequential(
            nn.Conv1d(proj_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        self.cnn_k7 = nn.Sequential(
            nn.Conv1d(proj_dim, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        self.merge = nn.Linear(cnn_channels * 3, cnn_channels)
        self.act_merge = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.bell = BellPooling(seq_len)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, rnafm_feat):
        h = self.norm(rnafm_feat)
        h = self.act_proj(self.proj(h))
        h = h.permute(0, 2, 1)
        h3 = self.cnn_k3(h)
        h5 = self.cnn_k5(h)
        h7 = self.cnn_k7(h)
        h_cat = torch.cat([h3, h5, h7], dim=1)
        h_cat = h_cat.permute(0, 2, 1)
        h_cat = self.act_merge(self.merge(h_cat))
        h_cat = self.drop(h_cat)
        weights = self.bell()
        h_weighted = (h_cat * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        out = self.classifier(h_weighted)
        return out.squeeze(-1)


class KmerBiLSTMTower(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, onehot_feat):
        output, _ = self.lstm(onehot_feat)
        center_hidden = output[:, 20, :]
        out = self.classifier(center_hidden)
        return out.squeeze(-1)


class StructureRGCNTower(nn.Module):
    def __init__(self, node_dim=21, hidden_dim=64, num_relations=3, num_bases=3,
                 dropout=0.4, edge_dropout=0.1):
        super().__init__()
        self.edge_dropout = edge_dropout
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations,
                              num_bases=num_bases)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations,
                              num_bases=num_bases)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.readout_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def _drop_edges(self, edge_index, edge_type):
        if not self.training or self.edge_dropout <= 0:
            return edge_index, edge_type
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=edge_index.device) >= self.edge_dropout
        if keep_mask.sum() < 2:
            keep_mask[:2] = True
        return edge_index[:, keep_mask], edge_type[keep_mask]

    def forward(self, graph_data):
        x, edge_index, edge_type = graph_data.x, graph_data.edge_index, graph_data.edge_type
        batch = graph_data.batch
        center_indices = graph_data.center_indices
        edge_index, edge_type = self._drop_edges(edge_index, edge_type)
        h = self.input_proj(x)
        h_res = h
        h = self.rgcn1(h, edge_index, edge_type)
        h = self.norm1(h + h_res)
        h = self.act(h)
        h = self.drop(h)
        h_res = h
        h = self.rgcn2(h, edge_index, edge_type)
        h = self.norm2(h + h_res)
        h = self.act(h)
        h = self.drop(h)
        graph_vec = global_mean_pool(h, batch)
        center_vec = h[center_indices]
        readout = torch.cat([graph_vec, center_vec], dim=-1)
        readout = self.readout_proj(readout)
        out = self.classifier(readout)
        return out.squeeze(-1)