import copy
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
from .deformable_transformer import build_deforamble_transformer
from .bafe import BAFE
from .gvr import GVR
_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

class PolygonHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()
        self.box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        num_classes = cfg.MODEL.PEVFormer.NUM_CLASSES
        num_corners = cfg.MODEL.PEVFormer.NUM_CORNERS
        d_model = cfg.MODEL.PEVFormer.HIDDEN_DIM
        num_queries = cfg.MODEL.PEVFormer.NUM_CORNERS
        self.use_contour_enhancer = cfg.MODEL.PEVFormer.USE_BAFE
        self.contour_enhancer = None
        if self.use_contour_enhancer:
            pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
            warmup_iters = cfg.MODEL.PEVFormer.get('CONTOUR_WARMUP_ITERS', 5000)
            self.contour_enhancer = BAFE(in_channels=d_model, grid_size=cfg.MODEL.PEVFormer.BAFE.GRID_SIZE, scaling_factor=cfg.MODEL.PEVFormer.BAFE.SCALING_FACTOR, buffer_width=cfg.MODEL.PEVFormer.BAFE.BUFFER_WIDTH, num_heads=cfg.MODEL.PEVFormer.BAFE.NUM_HEADS, warmup_iters=warmup_iters)
        transformer = build_deforamble_transformer()
        self.transformer_head = GAHD(d_model, num_classes, num_corners, transformer, num_queries, aux_loss=cfg.MODEL.PEVFormer.DEEP_SUPERVISION, use_geo_refine=cfg.MODEL.PEVFormer.get('USE_GVR', False), geo_refine_num_heads=cfg.MODEL.PEVFormer.GVR.NUM_HEADS if cfg.MODEL.PEVFormer.get('USE_GVR', False) else 8, geo_refine_num_layers=cfg.MODEL.PEVFormer.GVR.NUM_LAYERS if cfg.MODEL.PEVFormer.get('USE_GVR', False) else 2)
        self.num_corners = num_corners
        self.use_focal = cfg.MODEL.PEVFormer.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.PEVFormer.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)
        if self.use_contour_enhancer and self.contour_enhancer is not None:
            pass
        geo_refine = self.transformer_head.geo_refine
        if geo_refine is not None:
            nn.init.zeros_(geo_refine.coord_refine_head[-1].weight)
            nn.init.zeros_(geo_refine.coord_refine_head[-1].bias)
            nn.init.zeros_(geo_refine.logit_refine_head.weight)
            nn.init.zeros_(geo_refine.logit_refine_head.bias)
            print('[GVRM-INIT] 输出头已零初始化 (coord_refine最后一层 + logit_refine)')

    @staticmethod
    def _init_box_pooler(cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple((1.0 / input_shape[k].stride for k in in_features))
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        in_channels = [input_shape[f].channels for f in in_features]
        assert len(set(in_channels)) == 1, in_channels
        box_pooler = ROIPooler(output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio, pooler_type=pooler_type)
        return box_pooler

    def forward(self, features, init_bboxes):
        output = self.transformer_head(features, init_bboxes, self.box_pooler, self.contour_enhancer if self.use_contour_enhancer else None)
        return output

class GAHD(nn.Module):

    def __init__(self, d_model, num_classes, num_corners, transformer, num_queries, scale_clamp: float=_DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0), aux_loss=False, with_poly_refine=True, use_geo_refine=False, geo_refine_num_heads=8, geo_refine_num_layers=2):
        super().__init__()
        hidden_dim = d_model
        self.num_queries = num_queries
        self.num_corners = num_corners
        self.num_classes = num_classes
        self.transformer = transformer
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.coords_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, 2)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.logit_embed = nn.Embedding(num_queries, 1)
        self.aux_loss = aux_loss
        self.with_poly_refine = with_poly_refine
        if self.with_poly_refine:
            self.class_embed = _get_clones(self.class_embed, transformer.decoder.num_layers)
            self.coords_embed = _get_clones(self.coords_embed, transformer.decoder.num_layers)
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.coords_embed = self.coords_embed
        self.use_geo_refine = use_geo_refine
        self.geo_refine = None
        if use_geo_refine:
            self.geo_refine = GVR(d_model=hidden_dim, num_heads=geo_refine_num_heads, num_refine_layers=geo_refine_num_layers)
            print(f'[GVRM] GeometricVertexRefinement enabled: heads={geo_refine_num_heads}, layers={geo_refine_num_layers}')
        self._reset_parameters()
        self.bbox_weights = bbox_weights
        self.scale_clamp = scale_clamp

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.tgt_embed.weight, std=0.01)
        nn.init.normal_(self.query_embed.weight, std=0.01)
        nn.init.normal_(self.logit_embed.weight, std=0.01)
        if self.geo_refine is not None:
            nn.init.zeros_(self.geo_refine.coord_refine_head[-1].weight)
            nn.init.zeros_(self.geo_refine.coord_refine_head[-1].bias)
            nn.init.zeros_(self.geo_refine.logit_refine_head.weight)
            nn.init.zeros_(self.geo_refine.logit_refine_head.bias)

    def forward(self, features, bboxes, pooler, contour_enhancer=None):
        N, nr_boxes = bboxes.shape[:2]
        proposal_boxes = [Boxes(bboxes[b]) for b in range(N)]
        roi_features = pooler(features, proposal_boxes)
        if torch.isnan(roi_features).any() or torch.isinf(roi_features).any():
            print('Warning: NaN or Inf detected in roi_features')
            roi_features = torch.nan_to_num(roi_features, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
        contour_map = None
        if contour_enhancer is not None:
            enhanced_features, contour_map = contour_enhancer(roi_features)
            if torch.isnan(enhanced_features).any() or torch.isinf(enhanced_features).any():
                print('Warning: NaN or Inf in enhanced_features, using original')
                enhanced_features = roi_features
            roi_features = enhanced_features
        srcs = [roi_features]
        masks = [torch.zeros([roi_features.shape[0], roi_features.shape[2], roi_features.shape[3]], device=roi_features.device, dtype=torch.bool)]
        pos = [build_positional_embedding(roi_features, masks[0], num_pos_feats=128)]
        query_embed = self.query_embed.weight
        tgt = self.tgt_embed.weight
        logit_embed = self.logit_embed.weight
        try:
            hs, init_reference_out, inter_references, inter_classes = self.transformer(bboxes=bboxes, srcs=srcs, masks=masks, pos_embeds=pos, query_embed=query_embed, tgt=tgt, logit_embed=logit_embed, tgt_masks=None)
        except Exception as e:
            print(f'Error in transformer forward: {e}')
            raise
        num_layers = hs.shape[0]
        outputs_coord = inter_references.view(num_layers, N, nr_boxes, self.num_queries, 2)
        outputs_class = inter_classes.view(num_layers, N, nr_boxes, self.num_queries, self.num_classes)
        if self.num_classes == 1:
            outputs_class = outputs_class.squeeze(-1)
        if torch.isnan(outputs_coord).any() or torch.isinf(outputs_coord).any():
            print('Warning: NaN or Inf in outputs_coord')
            outputs_coord = torch.nan_to_num(outputs_coord, nan=0.5, posinf=1.0, neginf=0.0)
        if torch.isnan(outputs_class).any() or torch.isinf(outputs_class).any():
            print('Warning: NaN or Inf in outputs_class')
            outputs_class = torch.nan_to_num(outputs_class, nan=0.0, posinf=10.0, neginf=-10.0)
        if self.use_geo_refine and self.geo_refine is not None:
            last_hs = hs[-1]
            last_coords_abs = inter_references[-1]
            last_logits = inter_classes[-1]
            coord_delta, logit_delta = self.geo_refine(last_hs, last_coords_abs, last_logits)
            refined_coords_abs = last_coords_abs + coord_delta
            refined_logits = last_logits + logit_delta
            refined_coords = refined_coords_abs.view(N, nr_boxes, self.num_queries, 2)
            refined_class = refined_logits.view(N, nr_boxes, self.num_queries, self.num_classes)
            if self.num_classes == 1:
                refined_class = refined_class.squeeze(-1)
            if torch.isnan(refined_coords).any() or torch.isinf(refined_coords).any():
                print('Warning: NaN/Inf in GVRM refined_coords, falling back to decoder output')
                refined_coords = outputs_coord[-1]
                refined_class = outputs_class[-1]
            out = {'pred_logits': refined_class, 'pred_coords': refined_coords}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        else:
            out = {'pred_logits': outputs_class[-1], 'pred_coords': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_coords': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList((nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_positional_embedding(x, mask, num_pos_feats=128, temperature=10000):
    assert mask is not None
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    eps = 1e-06
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos
