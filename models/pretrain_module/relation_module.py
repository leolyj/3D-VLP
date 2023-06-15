"""
Modified from: https://github.com/zlccccc/3DVL_Codebase/blob/main/models/proposal_module/relation_module.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward
import random

from models.base_module.ema_utils import copy_params, _momentum_update


class RelationModule(nn.Module):
    def __init__(self, num_proposals=256, hidden_size=128, lang_num_size=300, det_channel=128, head=4, depth=2):
        super().__init__()
        self.use_box_embedding = True
        self.use_dist_weight_matrix = True
        self.use_obj_embedding = True
        # self.use_obj_embedding = False  # 3D only

        self.num_proposals = num_proposals
        self.hidden_size = hidden_size
        self.depth = depth

        self.features_concat = nn.Sequential(
            nn.Conv1d(det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )
        self.self_attn_fc = nn.ModuleList(
            nn.Sequential(  # 4 128 256 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 4)
        ) for i in range(depth))

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))
        self.self_attn_m = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))

        self.bbox_embedding = nn.ModuleList(nn.Linear(27, hidden_size) for i in range(depth))
        self.obj_embedding = nn.ModuleList(nn.Linear(128, hidden_size) for i in range(depth))

        self.bbox_proj = nn.Linear(128, 128)

        self.model_pairs = [[self.self_attn, self.self_attn_m]]
        copy_params(self.model_pairs)

    def _get_bbox_centers(self, corners):
        coord_min = torch.min(corners, dim=2)[0] # batch_size, num_proposals, 3
        coord_max = torch.max(corners, dim=2)[0] # batch_size, num_proposals, 3
        return (coord_min + coord_max) / 2

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        mask = data_dict['mask_label']
        B = mask.shape[0]

        # object size embedding
        features = data_dict['pred_bbox_feature'].permute(0, 2, 1)

        features = self.features_concat(features).permute(0, 2, 1)
        batch_size, num_proposal = features.shape[:2]
        dim_f = features.shape[-1]

        # Mask on 3D proposals
        features_vis = features[~mask].reshape(B, -1, dim_f)

        for i in range(self.depth):
            # relation emb
            if self.use_dist_weight_matrix:
                # Attention Weight
                # objects_center = data_dict['center']
                objects_center = data_dict['pred_bbox_corner'].mean(dim=-2)
                N_K = objects_center.shape[1]
                center_A = objects_center[:, None, :, :].repeat(1, N_K, 1, 1)
                center_B = objects_center[:, :, None, :].repeat(1, 1, N_K, 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, None, :, :]

                weights = torch.cat([center_dist, dist.permute(0, 2, 3, 1)], dim=-1).detach()  # K N N 4
                dist_weights = self.self_attn_fc[i](weights).permute(0, 3, 1, 2)

                attention_matrix_way = 'add'
            else:
                dist_weights = None
                attention_matrix_way = 'mul'

            # multiview/rgb feature embedding
            if self.use_obj_embedding:
                obj_feat = data_dict["point_clouds"][..., 6:6 + 128].permute(0, 2, 1)
                obj_feat_dim = obj_feat.shape[1]
                obj_feat_id_seed = data_dict["seed_inds"]
                obj_feat_id_seed = obj_feat_id_seed.long() + (
                    (torch.arange(batch_size) * obj_feat.shape[1])[:, None].to(obj_feat_id_seed.device))
                obj_feat_id_seed = obj_feat_id_seed.reshape(-1)
                obj_feat_id_vote = data_dict["aggregated_vote_inds"]
                obj_feat_id_vote = obj_feat_id_vote.long() + (
                    (torch.arange(batch_size) * data_dict["seed_inds"].shape[1])[:, None].to(
                        obj_feat_id_vote.device))
                obj_feat_id_vote = obj_feat_id_vote.reshape(-1)
                obj_feat_id = obj_feat_id_seed[obj_feat_id_vote]
                obj_feat = obj_feat.reshape(-1, obj_feat_dim)[obj_feat_id].reshape(batch_size, num_proposal,
                                                                                   obj_feat_dim)
                obj_embedding = self.obj_embedding[i](obj_feat)
                features = features + obj_embedding * 0.1

                dim_o = obj_embedding.shape[-1]
                features_vis = features_vis + obj_embedding[~mask].reshape(B, -1, dim_o) * 0.1

            # box embedding
            if self.use_box_embedding:
                corners = data_dict['pred_bbox_corner']
                centers = self._get_bbox_centers(corners)
                num_proposals = centers.shape[1]
                # attention weight
                manual_bbox_feat = torch.cat(
                    [centers, (corners - centers[:, :, None, :]).reshape(batch_size, num_proposals, -1)],
                    dim=-1).float()
                bbox_embedding = self.bbox_embedding[i](manual_bbox_feat)
                features = features + bbox_embedding

                dim_b = bbox_embedding.shape[-1]
                features_vis = features_vis + bbox_embedding[~mask].reshape(B, -1, dim_b)

                mask_bbox_embedding = bbox_embedding[mask].reshape(B, -1, dim_b)
                data_dict['mask_bbox_embedding'] = mask_bbox_embedding

                data_dict['manual_bbox_feat'] = manual_bbox_feat

            features_att = self.self_attn[i](features, features, features, attention_weights=dist_weights,
                                             way=attention_matrix_way)

            with torch.no_grad():
                _momentum_update(self.model_pairs, momentum=0.995)
                features_m = self.self_attn_m[i](features, features, features, attention_weights=dist_weights,
                                                 way=attention_matrix_way)

            dist_weights = dist_weights.permute(0, 2, 3, 1)
            d1, h = dist_weights.shape[2:4]
            dist_weights_vis = dist_weights[~mask].reshape(B, -1, d1, h)
            dist_weights_vis = dist_weights_vis.permute(0, 2, 1, 3)
            d2, h = dist_weights_vis.shape[2:4]
            dist_weights_vis = dist_weights_vis[~mask].reshape(B, -1, d2, h).permute(0, 3, 2, 1)

            # self-attention on visible proposal features
            features_vis = self.self_attn[i](features_vis, features_vis, features_vis, attention_weights=dist_weights_vis,
                                             way=attention_matrix_way)

        data_dict['dist_weights'] = dist_weights.permute(0, 3, 1, 2)
        data_dict['attention_matrix_way'] = attention_matrix_way

        data_dict["bbox_feature"] = features_att
        data_dict["bbox_feature_m"] = features_m  # momentum-updated
        data_dict["bbox_feature_vis"] = features_vis  # masked

        data_dict["bbox_proj_emb"] = F.normalize(self.bbox_proj(features_att), dim=-1)

        return data_dict


