import torch
import torch.nn as nn
import torch.nn.functional as F

class GVR(nn.Module):

    def __init__(self, d_model=256, num_heads=8, num_refine_layers=2):
        super().__init__()
        self.d_model = d_model
        self.geo_encoder = nn.Sequential(nn.Linear(5, d_model // 2), nn.ReLU(inplace=True), nn.Linear(d_model // 2, d_model), nn.LayerNorm(d_model))
        self.fusion = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(inplace=True), nn.LayerNorm(d_model))
        self.refine_layers = nn.ModuleList()
        for _ in range(num_refine_layers):
            self.refine_layers.append(nn.ModuleDict({'self_attn': nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1), 'norm1': nn.LayerNorm(d_model), 'ffn': nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(d_model * 2, d_model), nn.Dropout(0.1)), 'norm2': nn.LayerNorm(d_model)}))
        self.coord_refine_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 2))
        self.logit_refine_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.coord_refine_head[-1].weight)
        nn.init.zeros_(self.coord_refine_head[-1].bias)
        nn.init.zeros_(self.logit_refine_head.weight)
        nn.init.zeros_(self.logit_refine_head.bias)

    def compute_geometric_features(self, coords):
        B, M, _ = coords.shape
        prev_coords = torch.roll(coords, 1, dims=1)
        next_coords = torch.roll(coords, -1, dims=1)
        edge_prev = coords - prev_coords
        edge_next = next_coords - coords
        len_prev = torch.norm(edge_prev, dim=-1, keepdim=True).clamp(min=1e-06)
        len_next = torch.norm(edge_next, dim=-1, keepdim=True).clamp(min=1e-06)
        edge_prev_n = edge_prev / len_prev
        edge_next_n = edge_next / len_next
        neg_edge_prev_n = -edge_prev_n
        cos_angle = (neg_edge_prev_n * edge_next_n).sum(dim=-1, keepdim=True)
        sin_angle = (neg_edge_prev_n[..., 0] * edge_next_n[..., 1] - neg_edge_prev_n[..., 1] * edge_next_n[..., 0]).unsqueeze(-1)
        mean_len = torch.cat([len_prev, len_next], dim=-1).mean(dim=(1, 2), keepdim=True).clamp(min=1e-06)
        len_prev_norm = len_prev / mean_len
        len_next_norm = len_next / mean_len
        len_ratio = torch.log((len_prev / len_next).clamp(0.01, 100.0))
        geo_features = torch.cat([cos_angle, sin_angle, len_prev_norm, len_next_norm, len_ratio], dim=-1)
        return geo_features

    def forward(self, decoder_hs, pred_coords, pred_logits):
        with torch.no_grad():
            geo_raw = self.compute_geometric_features(pred_coords.detach())
        geo_feats = self.geo_encoder(geo_raw)
        x = torch.cat([decoder_hs, geo_feats], dim=-1)
        x = self.fusion(x)
        for layer in self.refine_layers:
            residual = x
            x2, _ = layer['self_attn'](x, x, x)
            x = layer['norm1'](residual + x2)
            residual = x
            x2 = layer['ffn'](x)
            x = layer['norm2'](residual + x2)
        coord_delta = self.coord_refine_head(x)
        logit_delta = self.logit_refine_head(x)
        return (coord_delta, logit_delta)
