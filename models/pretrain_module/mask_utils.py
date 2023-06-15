import torch
import torch.nn as nn
import numpy as np
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes


def generate_mask(device, mask_ratio=0.5, n_sample=256, B=10):
    overall_mask = np.zeros([B, n_sample])
    num_mask = int(n_sample * mask_ratio)
    for i in range(B):
        mask = np.hstack([
            np.zeros(n_sample - num_mask),
            np.ones(num_mask)
        ])
        np.random.shuffle(mask)
        overall_mask[i, :] = mask
    overall_mask = torch.from_numpy(overall_mask).to(torch.bool).to(device)

    return overall_mask


def proposal_mask(data_dict):
    proposal_xyz = data_dict['aggregated_vote_xyz']
    proposal_features = data_dict['aggregated_vote_features']  # (B, num_proposal, 128)
    vote_inds = data_dict['aggregated_vote_inds']
    # data_dict['pred_bbox_corner']
    mask = data_dict['mask_label']  # (B, num_proposal)

    B, n_proposal = proposal_xyz.shape[:2]

    vis_xyz = proposal_xyz[~mask].reshape(B, -1, 3)
    mask_xyz = proposal_xyz[mask].reshape(B, -1, 3)
    vis_features = proposal_features[~mask].reshape(B, -1, -128)  #############################

    vis_vote_inds = vote_inds[~mask].reshape(B, -1)
    mask_vote_inds = vote_inds[mask].reshape(B, -1)

    n_total_points = data_dict['point_clouds'].shape[1]  # 40000

    vis_inds = vis_vote_inds.long()
    vis_inds = vis_inds + ((torch.arange(B) * n_total_points)[:, None].to(vis_inds.device))
    vis_inds = vis_inds.reshape(-1)
    vis_sem = data_dict['sem_labels'].reshape(-1)[vis_inds]
    vis_sem = vis_sem.reshape(B, -1)

    mask_inds = mask_vote_inds.long()
    mask_inds = mask_inds + ((torch.arange(B) * n_total_points)[:, None].to(mask_inds.device))
    mask_inds = mask_inds.reshape(-1)
    mask_sem = data_dict['sem_labels'].reshape(-1)[mask_inds]
    mask_sem = mask_sem.reshape(B, -1)

    data_dict['mask_sem'] = mask_sem
    data_dict['cat_sem'] = torch.cat([vis_sem, mask_sem], dim=1)

    # pos_features = self.position_embedding(vis_xyz)

    # vis_inds = data_dict['fp2_inds'][:, 0:self.npoint][~mask].reshape(B, -1)
    # mask_inds = data_dict['fp2_inds'][:, 0:self.npoint][mask].reshape(B, -1)

    data_dict['vis_xyz'] = vis_xyz
    data_dict['vis_features'] = vis_features
    data_dict['vis_inds'] = vis_vote_inds
    data_dict['mask_xyz'] = mask_xyz
    data_dict['mask_inds'] = mask_vote_inds

    return data_dict


class EmbeddingModule(nn.Module):
    def __init__(self, npoint, feat_dim):
        super().__init__()
        self.npoint = npoint

        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=npoint,
            radius=0.3,
            nsample=16,
            mlp=[feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        # self.position_embedding = nn.Sequential(
        #     nn.Linear(3, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 128)
        # )

    def forward(self, data_dict, xyz, features, mask):
        B = xyz.shape[0]
        xyz, features, fps_inds = self.vote_aggregation(xyz, features)  # 1024 -> 256

        vis_xyz = xyz[~mask].reshape(B, -1, 3)
        mask_xyz = xyz[mask].reshape(B, -1, 3)
        vis_features = features.transpose(1, 2).contiguous()[~mask].reshape(B, -1, 128)

        # pos_features = self.position_embedding(vis_xyz)

        vis_inds = data_dict['fp2_inds'][:, 0:self.npoint][~mask].reshape(B, -1)
        mask_inds = data_dict['fp2_inds'][:, 0:self.npoint][mask].reshape(B, -1)

        data_dict['vis_xyz'] = vis_xyz
        data_dict['vis_features'] = vis_features
        data_dict['vis_inds'] = vis_inds
        data_dict['mask_xyz'] = mask_xyz
        data_dict['mask_inds'] = mask_inds

        return data_dict
