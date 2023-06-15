import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward
import random
import timm
from timm.models.layers import trunc_normal_
import numpy as np

from models.base_module.ema_utils import copy_params, _momentum_update


class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, head=4, depth=2, mask_class=18,
                 vocab_size=3433, recon_points=False):
        super().__init__()
        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.recon_points = recon_points

        # proposal-language matching decoder
        self.match = build_layers(hidden_size, 1)

        # not used during pretraining
        self.grounding_cross_attn = MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)  # k, q, v

        self.bbox_embedding = nn.Linear(27, hidden_size)

        self.self_att = mhatt(hidden_size=hidden_size, head=head, depth=depth)
        self.cross_att = mhatt(hidden_size=hidden_size, head=head, depth=depth)
        self.self_att_m = mhatt(hidden_size=hidden_size, head=head, depth=depth)
        self.cross_att_m = mhatt(hidden_size=hidden_size, head=head, depth=depth)

        self.model_paris = [[self.self_att, self.self_att_m], [self.cross_att, self.cross_att_m]]
        copy_params(self.model_paris)

        # for MLM
        self.lang_self_att = mhatt(hidden_size=hidden_size, head=head, depth=depth)
        self.lang_cross_att = mhatt(hidden_size=hidden_size, head=head, depth=depth)

        # decoder for masked proposal
        self.cls_head = build_layers(hidden_size, mask_class)

        if recon_points:
            self.recon_p_head = build_layers(hidden_size, 3 * 64)

        # decoder for masked language
        self.mlm_head = build_layers(hidden_size, vocab_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size)).cuda()

        trunc_normal_(self.mask_token, std=.02)

    def forward(self, data_dict):
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1
        features = data_dict["bbox_feature"]  # batch_size, num_proposals, feat_size

        batch_size, num_proposal = features.shape[:2]
        len_nun_max = data_dict["lang_feat_list"].shape[1]
        data_dict["random"] = random.random()

        # copy paste
        feature0 = features.clone()
        if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
            obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
            obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == True)[0]
                obj_len = obj_mask.shape[0]
                obj_lens[i] = obj_len

            obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
            obj_features = features.reshape(batch_size*num_proposal, -1)
            obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
            total_len = obj_mask.shape[0]
            obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
            j = 0
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == False)[0]
                obj_len = obj_mask.shape[0]
                j += obj_lens[i]
                if obj_len < total_len - obj_lens[i]:
                    feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                else:
                    feature0[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]

        proposal_fea = feature0[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, num_proposal, -1)
        lang_fea = data_dict["lang_fea"]

        # ---------------------------------- MPM ----------------------------------
        mask = data_dict['mask_label']
        vis_features = data_dict['bbox_feature_vis']

        manual_bbox_feat = data_dict['manual_bbox_feat']
        bbox_pos_embedding = self.bbox_embedding(manual_bbox_feat)
        vis_pos_embedding = bbox_pos_embedding[~mask].reshape(batch_size, -1, self.hidden_size)
        mask_pos_embedding = bbox_pos_embedding[mask].reshape(batch_size, -1, self.hidden_size)

        mask_features = self.mask_token.repeat(mask_pos_embedding.shape[0], mask_pos_embedding.shape[1], 1)
        # mask_token + position_embedding
        mask_features = mask_features + mask_pos_embedding
        vis_features = vis_features + vis_pos_embedding  # 9/20

        # concat mask_tokens
        features_cat = torch.cat([vis_features, mask_features], dim=1)

        batch_size, num_patches = features_cat.shape[:2]
        len_nun_max = data_dict["lang_feat_list"].shape[1]

        # copy paste
        feature0_m = features_cat.clone()  # (B, 256, 128)

        feature1_m = feature0_m[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, num_patches, -1)
        lang_fea_full = data_dict["lang_fea_full"]  # unmasked

        # cross-attention
        for n in range(self.depth):
            feature1_m = self.cross_att[n](feature1_m, lang_fea_full, lang_fea_full, data_dict["attention_mask"])

        # masked proposal cls prediction
        data_dict["bbox_features_mask"] = feature1_m
        feature1_agg_m = feature1_m.permute(0, 2, 1).contiguous()

        pred_cls = self.cls_head(feature1_agg_m).permute(0, 2, 1)
        data_dict['pred_cls'] = pred_cls

        if self.recon_points:
            pred_points = self.recon_p_head(feature1_agg_m).permute(0, 2, 1)
            data_dict['pred_points'] = pred_points

        # momentum-updated branch
        with torch.no_grad():
            _momentum_update(self.model_paris, momentum=0.995)
            features_moment = data_dict["bbox_feature_m"]
            features_moment = features_moment[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size * len_nun_max, num_proposal, -1)

            for n in range(self.depth):
                features_moment = self.cross_att_m[n](features_moment, lang_fea_full, lang_fea_full, data_dict["attention_mask"])

            data_dict["bbox_features_moment"] = features_moment

        # ---------------------------------- MLM ----------------------------------
        feature1 = lang_fea
        for n in range(self.depth):
            feature1 = self.lang_cross_att[n](feature1, proposal_fea, proposal_fea)

        feature1 = feature1.permute(0, 2, 1).contiguous()
        pred_lang_mask = self.mlm_head(feature1)
        pred_lang_mask = pred_lang_mask.permute(0, 2, 1).contiguous()
        data_dict["pred_lang_mask"] = pred_lang_mask

        # -------------------------------- Matching --------------------------------
        feature2 = proposal_fea
        # cross-attention
        for n in range(self.depth):
            feature2 = self.cross_att[n](feature2, lang_fea, lang_fea, data_dict["attention_mask"])

        # match
        feature2_agg = feature2
        feature2_agg = feature2_agg.permute(0, 2, 1).contiguous()

        confidence1 = self.match(feature2_agg).squeeze(1)  # batch_size, num_proposals
        data_dict["cluster_ref"] = confidence1

        return data_dict


def build_layers(hidden_size, out_size):

    return nn.Sequential(
        nn.Conv1d(hidden_size, hidden_size, 1),
        nn.BatchNorm1d(hidden_size),
        nn.PReLU(),
        nn.Conv1d(hidden_size, hidden_size, 1),
        nn.BatchNorm1d(hidden_size),
        nn.PReLU(),
        nn.Conv1d(hidden_size, out_size, 1)
    )


def mhatt(hidden_size, head, depth):

    return nn.ModuleList(
        MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in
        range(depth)
    )
