import copy
from typing import Optional, List
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from .utils import inverse_sigmoid
from .ops.modules import MSDeformAttn

class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, activation='relu', poly_refine=True, return_intermediate_dec=False, aux_loss=False, num_feature_levels=4, dec_n_points=4, enc_n_points=4, query_pos_type='none'):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, poly_refine, return_intermediate_dec, aux_loss, query_pos_type)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        if query_pos_type == 'sine':
            self.decoder.pos_trans = nn.Linear(d_model, d_model)
            self.decoder.pos_trans_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, bboxes, srcs, masks, pos_embeds, query_embed=None, tgt=None, logit_embed=None, tgt_masks=None):
        assert query_embed is not None
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        bs, _, c = memory.shape
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = query_embed.sigmoid()
        init_reference_out = reference_points
        logit_embed = logit_embed.unsqueeze(0).expand(bs, -1, -1)
        hs, inter_references, inter_classes = self.decoder(bboxes, tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, logit_embed, query_pos=query_embed, src_padding_mask=mask_flatten, tgt_masks=tgt_masks)
        return (hs, init_reference_out, inter_references, inter_classes)

class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src

class DeformableTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device), torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output

class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.block_logit_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def with_logit_embed(self, tensor, logit_emb):
        N, num_queries = (tensor.shape[0], tensor.shape[1])
        scale_shift = self.block_logit_mlp(logit_emb)
        scale_shift = scale_shift.flatten(0, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        tensor = tensor.flatten(0, 1)
        tensor = tensor * (scale + 1) + shift
        tensor = tensor.view(N, num_queries, -1)
        return tensor

    def forward(self, tgt, query_pos, logit_embed, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, tgt_masks=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        q = k = self.with_logit_embed(q, logit_embed)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=tgt_masks)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.cross_attn(self.with_logit_embed(self.with_pos_embed(tgt, query_pos), logit_embed), reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)
        return tgt

class DeformableTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, poly_refine=True, return_intermediate=False, aux_loss=False, query_pos_type='none'):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.poly_refine = poly_refine
        self.logit_refine = True
        self.return_intermediate = return_intermediate
        self.aux_loss = aux_loss
        self.query_pos_type = query_pos_type
        self.coords_embed = None
        self.class_embed = None
        self.pos_trans = None
        self.pos_trans_norm = None
        self.d_model = 256
        logit_dim = self.d_model * 4
        self.logit_mlp = nn.Sequential(SinusoidalPositionEmbeddings(self.d_model), nn.Linear(self.d_model, logit_dim), nn.GELU(), nn.Linear(logit_dim, logit_dim))

    def get_query_pos_embed(self, ref_points):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ref_points.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        ref_points = ref_points * scale
        pos = ref_points[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(self, bboxes, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, logit_embed, query_pos=None, src_padding_mask=None, tgt_masks=None):
        output = tgt
        _, num_queries, _ = tgt.shape
        x1y1 = bboxes.flatten(0, 1)[:, 0:2]
        x2y2 = bboxes.flatten(0, 1)[:, 2:]
        wh = x2y2 - x1y1
        x1y1 = x1y1[:, None, :].expand(-1, num_queries, -1)
        wh = wh[:, None, :].expand(-1, num_queries, -1)
        intermediate = []
        intermediate_reference_points = []
        intermediate_classes = []
        for lid, layer in enumerate(self.layers):
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.query_pos_type == 'sine':
                query_pos = self.pos_trans_norm(self.pos_trans(self.get_query_pos_embed(reference_points)))
            elif self.query_pos_type == 'none':
                query_pos = None
            logit_embedding = self.logit_mlp(logit_embed.sigmoid())
            output = layer(output, query_pos, logit_embedding, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, tgt_masks)
            if self.poly_refine:
                offset = self.coords_embed[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset + inverse_sigmoid(reference_points)
                reference_points = new_reference_points.sigmoid()
            elif lid == len(self.layers) - 1:
                offset = self.coords_embed[-1](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset + inverse_sigmoid(reference_points)
                reference_points = new_reference_points.sigmoid()
            if self.logit_refine:
                logit_offset = self.class_embed[lid](output)
                logit_embed = logit_embed + logit_offset
            elif lid == len(self.layers) - 1:
                logit_offset = self.class_embed[-1](output)
                logit_embed = logit_embed + logit_offset
            global_reference_points = reference_points * wh + x1y1
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(global_reference_points)
                intermediate_classes.append(logit_embed)
        if self.return_intermediate:
            return (torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_classes))
        return (output, global_reference_points, logit_embed)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')

class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, logits):
        device = logits.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = logits * embeddings[None, None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def build_deforamble_transformer():
    return DeformableTransformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, activation='relu', poly_refine=True, return_intermediate_dec=True, aux_loss=False, num_feature_levels=1, dec_n_points=4, enc_n_points=4, query_pos_type='sine')
