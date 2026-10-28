import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BAFE(nn.Module):

    def __init__(self, in_channels: int=256, grid_size: int=7, scaling_factor: float=4.0, buffer_width: int=2, num_heads: int=8, warmup_iters: int=5000):
        super(BAFE, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.scaling_factor = scaling_factor
        self.buffer_width = buffer_width
        self.warmup_iters = max(1, warmup_iters)
        self.current_iter = 0
        self.sparse_feature_extractor = BGS(in_channels, grid_size, scaling_factor)
        self.dense_feature_extractor = CBL(in_channels, num_heads)
        self.feature_fusion = AFF(in_channels, buffer_width)

    def forward(self, roi_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = 1.0
        sparse_features, deformed_points = self.sparse_feature_extractor(roi_features)
        dense_features, contour_map = self.dense_feature_extractor(roi_features, sparse_features)
        enhanced_features = self.feature_fusion(roi_features, sparse_features, dense_features, contour_map, deformed_points, alpha)
        if torch.isnan(enhanced_features).any() or torch.isinf(enhanced_features).any():
            print(f'[WARNING] NaN/Inf in enhanced_features at iter {self.current_iter}, using original')
            enhanced_features = roi_features
            contour_map = torch.zeros_like(contour_map)
        return (enhanced_features, contour_map)

    def reset_warmup(self):
        self.current_iter = 0

class BGS(nn.Module):

    def __init__(self, in_channels: int, grid_size: int, scaling_factor: float):
        super(BGS, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.scaling_factor = min(scaling_factor / grid_size, 0.7)
        self.query_proj = nn.Linear(in_channels, in_channels)
        self.edge_prior = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU())
        self.offset_network = nn.Sequential(nn.Conv2d(in_channels + 32, in_channels, kernel_size=1, bias=False), nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels), nn.GroupNorm(32, in_channels), nn.GELU(), nn.Conv2d(in_channels, 2, kernel_size=1), nn.Tanh())
        nn.init.xavier_uniform_(self.offset_network[-2].weight, gain=0.01)
        nn.init.zeros_(self.offset_network[-2].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, self.grid_size, device=x.device), torch.linspace(-1, 1, self.grid_size, device=x.device))
        grid_points = torch.stack([grid_x, grid_y], dim=-1)
        grid_points = grid_points.unsqueeze(0).repeat(B, 1, 1, 1)
        x_flat = x.flatten(2).permute(0, 2, 1)
        query = self.query_proj(x_flat)
        query = query.permute(0, 2, 1).reshape(B, C, H, W)
        edge_feat = self.edge_prior(x)
        query_with_edge = torch.cat([query, edge_feat], dim=1)
        offset = self.offset_network(query_with_edge)
        if H != self.grid_size or W != self.grid_size:
            offset_resized = F.interpolate(offset, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=True)
        else:
            offset_resized = offset
        offset_scaled = offset_resized * self.scaling_factor
        offset_scaled = offset_scaled.permute(0, 2, 3, 1)
        offset_scaled = torch.clamp(offset_scaled, -0.7, 0.7)
        deformed_points = grid_points + offset_scaled
        deformed_points = torch.clamp(deformed_points, -1, 1)
        sparse_features = F.grid_sample(x, deformed_points, mode='bilinear', padding_mode='zeros', align_corners=True)
        sparse_features = sparse_features.flatten(2)
        deformed_points = deformed_points.reshape(B, -1, 2)
        return (sparse_features, deformed_points)

class CBL(nn.Module):

    def __init__(self, in_channels: int, num_heads: int=8):
        super(CBL, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        self.confidence_gate_linear = nn.Linear(in_channels, 1)
        nn.init.zeros_(self.confidence_gate_linear.weight)
        nn.init.constant_(self.confidence_gate_linear.bias, 4.0)
        self.contour_segmenter = ContourSegmenter(in_channels)
        self._last_gate_mean = None

    def forward(self, feature_map: torch.Tensor, sparse_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = feature_map.shape
        query = feature_map.flatten(2).permute(0, 2, 1)
        sparse_features_t = sparse_features.permute(0, 2, 1)
        gate = torch.sigmoid(self.confidence_gate_linear(sparse_features_t))
        sparse_gated = sparse_features_t * gate
        key = self.key_proj(sparse_gated)
        value = self.value_proj(sparse_gated)
        attn_output, _ = self.multihead_attention(query=query, key=key, value=value)
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)
        contour_map = self.contour_segmenter(attn_output)
        contour_weight = torch.sigmoid(contour_map)
        dense_features = attn_output * (1 + 0.5 * contour_weight)
        return (dense_features, contour_map)

class ContourSegmenter(nn.Module):

    def __init__(self, in_channels: int):
        super(ContourSegmenter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(32, in_channels)
        self.dropout = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=0.1)
        nn.init.constant_(self.conv2.bias, -0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.conv1(features)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class AFF(nn.Module):

    def __init__(self, in_channels: int, buffer_width: int):
        super(AFF, self).__init__()
        self.in_channels = in_channels
        self.buffer_width = min(buffer_width, 2)
        self.attention = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.Sigmoid())
        self._residual_scale = nn.Parameter(torch.ones(1))
        self.register_buffer('ema_scale', torch.tensor(1.0))
        self.ema_momentum = 0.99
        self.contour_sharpness = nn.Parameter(torch.ones(1))
        self._last_sharpness = None

    @property
    def residual_scale(self):
        return 2.0 * torch.tanh(self._residual_scale)

    def forward(self, original_features: torch.Tensor, sparse_features: torch.Tensor, dense_features: torch.Tensor, contour_map: torch.Tensor, deformed_points: torch.Tensor, alpha: float=1.0) -> torch.Tensor:
        B, C, N = sparse_features.shape
        _, _, H, W = dense_features.shape
        grid_size = int(N ** 0.5)
        contour_buffer = self._create_contour_buffer(contour_map, self.buffer_width)
        filtered_dense = dense_features * (1 + 0.3 * contour_buffer)
        sampled_dense = F.grid_sample(filtered_dense, deformed_points.reshape(B, grid_size, grid_size, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        sparse_reshaped = sparse_features.reshape(B, C, grid_size, grid_size)
        if grid_size != H or grid_size != W:
            sparse_reshaped = F.interpolate(sparse_reshaped, size=(H, W), mode='bilinear', align_corners=True)
            sampled_dense = F.interpolate(sampled_dense, size=(H, W), mode='bilinear', align_corners=True)
        combined = torch.cat([sparse_reshaped, sampled_dense], dim=1)
        attention_weight = self.attention(combined)
        enhanced = attention_weight * (sparse_reshaped + sampled_dense)
        scale = self.residual_scale
        enhanced_scaled = alpha * scale * enhanced
        enhanced_scaled = torch.clamp(enhanced_scaled, -15, 15)
        fused_features = original_features + enhanced_scaled
        return fused_features

    def _create_contour_buffer(self, contour_map: torch.Tensor, buffer_width: int) -> torch.Tensor:
        if buffer_width == 0:
            return torch.ones_like(contour_map)
        temperature = torch.clamp(self.contour_sharpness, 0.1, 10.0)
        soft_contour = torch.sigmoid(contour_map * temperature)
        kernel_size = 2 * buffer_width + 1
        padding = buffer_width
        dilated = F.max_pool2d(soft_contour, kernel_size=kernel_size, stride=1, padding=padding)
        if dilated.shape != contour_map.shape:
            dilated = F.interpolate(dilated, size=contour_map.shape[-2:], mode='nearest')
        return dilated
