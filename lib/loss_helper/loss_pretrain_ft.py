import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import copy
from lib.configs.config_pretrain import CONF
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
from .loss_detection import compute_vote_loss, compute_objectness_loss, compute_box_loss, compute_box_and_sem_cls_loss
from .loss_captioning import compute_cap_loss
from .loss_grounding import compute_reference_loss, compute_lang_classification_loss
from utils.nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.3
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.point_criterion = nn.L1Loss(reduction='mean')

    def forward(self, x1, y1):

        x = x1.unsqueeze(1)
        y = y1.unsqueeze(2)
        dist = torch.sqrt(1e-8 + torch.sum(torch.pow(x - y, 2), 3))  # bs, ny, nx --- pointwise dist
        min1, _ = torch.min(dist, 1)
        min2, _ = torch.min(dist, 2)

        return min1.mean() + min2.mean()


# semantic and contextual alignment
def compute_align_loss_2(data_dict):
    align_label = data_dict['align_labels']

    temp = data_dict["temp"]
    proposal_fea = data_dict['proposal_proj_emb']
    bbox_feature = data_dict['bbox_proj_emb']

    lang_fea = data_dict['lang_proj_fea']
    lang_emb = data_dict['lang_proj_emb']

    batch_size, n_proposal = proposal_fea.shape[:2]
    n_lang = lang_emb.shape[0] // batch_size

    dim = lang_emb.shape[-1]
    assert dim == proposal_fea.shape[-1], "feature dim mismatch between proposal_fea and lang_emb"
    assert dim == bbox_feature.shape[-1], "feature dim mismatch between bbox_feature and lang_emb"
    bbox_feature = bbox_feature.reshape(-1, dim)
    lang_emb = lang_emb.reshape(-1, dim)

    proposal_fea = proposal_fea[:, None, :, :].repeat(1, n_lang, 1, 1).reshape(batch_size*n_lang, n_proposal, -1)
    sim_proposal_lang = proposal_fea @ lang_fea.transpose(1, 2) / temp

    sim_bbox_lang = bbox_feature @ lang_emb.t() / temp
    align_label = align_label.transpose(1, 2)

    sim_targets_pl = torch.zeros(sim_proposal_lang.size()).to(sim_proposal_lang.device)
    first_obj = data_dict["ground_first_obj_list"].reshape(batch_size * n_lang)
    first_obj_proposal = data_dict['align_labels'].reshape(batch_size * n_lang, -1)
    for j in range(sim_targets_pl.shape[0]):
        idx = torch.nonzero(first_obj_proposal[j] == 1)
        idx = idx.squeeze()
        sim_targets_pl[j, idx, first_obj[j]] = 1

    sim_targets_bl = torch.zeros(sim_bbox_lang.size()).to(sim_bbox_lang.device)
    for i in range(batch_size):
        sim_targets_bl[i*n_proposal:(i+1)*n_proposal, i*n_lang:(i+1)*n_lang] = align_label[i, :, :]

    loss_pl = -torch.sum(F.log_softmax(sim_proposal_lang, dim=1) * sim_targets_pl, dim=1).mean()
    loss_bl = -torch.sum(F.log_softmax(sim_bbox_lang, dim=1) * sim_targets_bl, dim=1).mean()

    loss = loss_bl + loss_pl

    return loss


def compute_mask_box_loss(data_dict, config):
    objectness_label = data_dict['objectness_label'].float()  # (B, N)
    batch_size = objectness_label.shape[0]
    num_proposal = objectness_label.shape[1]
    mask = data_dict['mask_label']  # (B, N)
    objectness_label_m = objectness_label[mask].reshape(batch_size, -1)
    num_mask = objectness_label_m.shape[1]

    gt_heading_class_m = data_dict['gt_assigned_heading_class'][mask].reshape(batch_size, -1)  # (B, N)
    gt_heading_residual_m = data_dict['gt_assigned_heading_residual'][mask].reshape(batch_size, -1)  # (B, N)
    gt_distance_m = data_dict['gt_assigned_distance'][mask].reshape(batch_size, -1, 6)  # (B, N, 6)

    # ------------- sem_cls_loss for mask -------------
    pred_cls = data_dict['pred_cls']  # (B*len_num_max, 256, 19)
    box_cls_label_m = data_dict['box_cls_label'][mask].reshape(batch_size, -1)

    len_nun_max = pred_cls.shape[0] // batch_size
    proposal_label = data_dict['proposal_label'].transpose(2, 1)  # (10, 8, 256) -> (10, 256, 8)
    proposal_label_mask = proposal_label[mask].reshape(batch_size, -1, len_nun_max)  # (10, 192, 8)
    proposal_label_mask = proposal_label_mask.transpose(2, 1).reshape(batch_size * len_nun_max, -1)  # (80, 192)

    objectness_label_m = objectness_label_m[:, None, :].repeat(1, len_nun_max, 1).reshape(batch_size*len_nun_max, -1)

    criterion_mask_cls = nn.CrossEntropyLoss(reduction='none')
    pred_cls_mask = pred_cls[:, (num_proposal-num_mask):, :]  # (80, 192, n_cls)
    mask_sem_cls_loss = criterion_mask_cls(pred_cls_mask.transpose(2, 1), box_cls_label_m[:, None, :].repeat(1, len_nun_max, 1).reshape(batch_size*len_nun_max, -1))
    mask_sem_cls_loss = torch.sum(mask_sem_cls_loss * objectness_label_m * proposal_label_mask) / (torch.sum(objectness_label_m * proposal_label_mask) + 1e-6)

    '''
    # ------------- heading_cls -------------
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    pred_heading_scores_m = data_dict['heading_scores_m'][:, (num_proposal-num_mask):, :]
    heading_class_loss_m = criterion_heading_class(pred_heading_scores_m.transpose(2, 1), gt_heading_class_m[:, None, :].repeat(1, len_nun_max, 1).reshape(batch_size*len_nun_max, -1))  # (B, N)
    heading_class_loss_m = torch.sum(heading_class_loss_m * objectness_label_m * proposal_label_mask)/(torch.sum(objectness_label_m * proposal_label_mask)+1e-6)

    # ------------- heading_residual -------------
    heading_residual_normalized_label_m = gt_heading_residual_m / (np.pi/1)  # num_heading_bin = 1
    heading_residual_normalized_label_m = heading_residual_normalized_label_m[: None, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, -1)
    heading_label_one_hot_m = torch.cuda.FloatTensor(batch_size, num_mask, 1).zero_()
    heading_label_one_hot_m.scatter_(2, gt_heading_class_m.unsqueeze(-1), 1)
    heading_label_one_hot_m = heading_label_one_hot_m[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, -1, 1)
    heading_residuals_normalized_m = data_dict['heading_residuals_normalized_m'][:, (num_proposal - num_mask):, :]
    heading_residual_normalized_loss_m = huber_loss(torch.sum(heading_residuals_normalized_m*heading_label_one_hot_m, -1) - heading_residual_normalized_label_m, delta=1.0)  # (B, N)
    heading_residual_normalized_loss_m = torch.sum(heading_residual_normalized_loss_m * objectness_label_m * proposal_label_mask)/(torch.sum(objectness_label_m * proposal_label_mask)+1e-6)

    # ------------- distance loss -------------
    rois_m = data_dict['pred_rois_m'][:, (num_proposal-num_mask):, :]
    gt_distance_m = gt_distance_m[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, -1, 6)
    distance_loss_m = torch.mean(huber_loss(rois_m - gt_distance_m, delta=0.15), -1)  # (B, N)
    distance_loss_m = torch.sum(distance_loss_m * objectness_label_m * proposal_label_mask)/(torch.sum(objectness_label_m * proposal_label_mask)+1e-6)
    '''

    heading_class_loss_m = torch.zeros(1)[0].to(pred_cls.device)
    heading_residual_normalized_loss_m = torch.zeros(1)[0].to(pred_cls.device)
    distance_loss_m = torch.zeros(1)[0].to(pred_cls.device)

    momnet_loss = torch.zeros(1)[0].to(pred_cls.device)
    if "bbox_features_moment" in data_dict:
        criterion_moment = nn.L1Loss()
        mask_feature = data_dict["bbox_features_mask"][:, (num_proposal-num_mask):, :]
        moment_feature = data_dict["bbox_features_moment"][:, (num_proposal-num_mask):, :]
        momnet_loss = criterion_moment(mask_feature, moment_feature)

    mask_sem_cls_loss = mask_sem_cls_loss + momnet_loss

    return heading_class_loss_m, heading_residual_normalized_loss_m, distance_loss_m, mask_sem_cls_loss


def compute_mask_lang_loss(data_dict, config):
    criterion = torch.nn.CrossEntropyLoss()

    lang_ids_list = data_dict['ground_lang_ids_list']
    batch_size, lang_num = lang_ids_list.shape[:2]
    lang_ids_list = lang_ids_list.reshape(batch_size*lang_num, -1)
    lang_len = data_dict["ground_lang_len_list"].reshape(-1)
    pred_lang_ids = data_dict["pred_lang_mask"]
    lang_mask_list = data_dict['lang_mask_list']

    lang_mask_ = torch.zeros(batch_size*lang_num, pred_lang_ids.shape[1]).to(pred_lang_ids.device)
    for i in range(batch_size*lang_num):
        masked_ids = lang_mask_list[i].to(torch.long)
        for ids in masked_ids:
            lang_mask_[i, ids] = 1
    num_mask = torch.sum(lang_mask_)

    gt_lang_ids = lang_ids_list[:, :pred_lang_ids.shape[1]]
    loss = criterion(pred_lang_ids.transpose(2, 1), gt_lang_ids)
    loss = torch.sum(loss * lang_mask_) / num_mask

    return loss


def get_pretrain_loss(data_dict, device, config, weights, detection=True, reference=True, use_lang_classifier=True,
                      orientation=False, distance=False, num_bins=CONF.TRAIN.NUM_BINS, num_epoch=150
                      ):
    # -------------- Detection loss --------------
    # Vote loss
    vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    data_dict["pos_ratio"] = torch.sum(objectness_label.float())/float(total_num_proposal)
    data_dict["neg_ratio"] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict["pos_ratio"]

    # Box loss and sem cls loss
    heading_cls_loss, heading_reg_loss, size_distance_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    box_loss = 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * sem_cls_loss
    box_loss = box_loss + 20 * size_distance_loss

    # objectness; Nothing
    obj_pred_val = torch.argmax(data_dict["objectness_scores"], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==data_dict["objectness_label"].long()).float()*data_dict["objectness_mask"])/(torch.sum(data_dict["objectness_mask"])+1e-6)
    data_dict["obj_acc"] = obj_acc

    if detection:
        data_dict["vote_loss"] = vote_loss
        data_dict["objectness_loss"] = objectness_loss
        data_dict["heading_cls_loss"] = heading_cls_loss
        data_dict["heading_reg_loss"] = heading_reg_loss
        data_dict["size_distance_loss"] = size_distance_loss
        data_dict["sem_cls_loss"] = sem_cls_loss
        data_dict["box_loss"] = box_loss
    else:
        device = vote_loss.device
        data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
        data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_distance_loss"] = torch.zeros(1)[0].to(device)
        data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["box_loss"] = torch.zeros(1)[0].to(device)

    if reference:
        # Reference loss
        data_dict, ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["ref_loss"] = ref_loss
    else:
        # raise NotImplementedError('Only detection; not implemented')
        # # Reference loss
        data_dict, ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config, no_reference=True)
        lang_count = data_dict['ref_center_label_list'].shape[1]
        # data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda().repeat(lang_count, 1)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda().repeat(lang_count, 1)
        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
        # data_dict['max_iou_rate_0.25'] = 0
        # data_dict['max_iou_rate_0.5'] = 0

    if reference and use_lang_classifier:
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    if orientation:
        raise NotImplementedError()
        ori_loss, ori_acc = compute_node_orientation_loss(data_dict, num_bins)

        # store
        data_dict["ori_loss"] = ori_loss
        data_dict["ori_acc"] = ori_acc
    else:
        # store
        data_dict["ori_loss"] = torch.zeros(1)[0].to(device)
        data_dict["ori_acc"] = torch.zeros(1)[0].to(device)

    if distance:
        raise NotImplementedError()
        dist_loss = compute_node_distance_loss(data_dict)

        # store
        data_dict["dist_loss"] = dist_loss
    else:
        # store
        data_dict["dist_loss"] = torch.zeros(1)[0].to(device)

    if detection:
        loss = data_dict["vote_loss"] + 0.1 * data_dict["objectness_loss"] + data_dict["box_loss"]
        loss *= 10  # amplify

        if orientation:
            loss += 0.1 * data_dict["ori_loss"]
        if distance:
            loss += 0.1 * data_dict["dist_loss"]
        if reference:
            loss += 0.3 * data_dict["ref_loss"]
        if use_lang_classifier:
            loss += 0.3 * data_dict["lang_loss"]
    else:
        raise NotImplementedError()
        loss = 0.

        if orientation:
            loss += 0.1 * data_dict["ori_loss"]
        if distance:
            loss += 0.1 * data_dict["dist_loss"]
        if reference:
            loss += 0.3 * data_dict["ref_loss"]
        if use_lang_classifier:
            loss += 0.3 * data_dict["lang_loss"]

    # -------------- Pretrain loss --------------
    # contrastive alignment loss
    align_loss = compute_align_loss_2(data_dict)

    data_dict['align_loss'] = align_loss
    loss += 5 * data_dict['align_loss']

    _, _, _, mask_sem_cls_loss = compute_mask_box_loss(data_dict, config)

    # masked bbox modeling
    data_dict['mpm_loss'] = mask_sem_cls_loss

    if data_dict["epoch"] < num_epoch:
        loss += 1 * data_dict['mpm_loss']

    mlm_loss = compute_mask_lang_loss(data_dict, config)
    data_dict['mlm_loss'] = 0.2 * mlm_loss

    if data_dict["epoch"] < num_epoch:
        loss += 1 * data_dict['mlm_loss']

    data_dict["loss"] = loss

    return data_dict


def get_ft_loss(data_dict, device, config, weights,
                detection=True, caption=True, reference=True, use_lang_classifier=True,
                orientation=False, distance=False, num_bins=CONF.TRAIN.NUM_BINS, num_ground_epoch=50, num_pretrain_epoch=0,
                ):
    # Vote loss
    vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    data_dict["pos_ratio"] = torch.sum(objectness_label.float())/float(total_num_proposal)
    data_dict["neg_ratio"] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict["pos_ratio"]

    # Box loss and sem cls loss
    heading_cls_loss, heading_reg_loss, size_distance_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    box_loss = 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * sem_cls_loss
    box_loss = box_loss + 20 * size_distance_loss

    # objectness; Nothing
    obj_pred_val = torch.argmax(data_dict["objectness_scores"], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==data_dict["objectness_label"].long()).float()*data_dict["objectness_mask"])/(torch.sum(data_dict["objectness_mask"])+1e-6)
    data_dict["obj_acc"] = obj_acc

    if detection:
        data_dict["vote_loss"] = vote_loss
        data_dict["objectness_loss"] = objectness_loss
        data_dict["heading_cls_loss"] = heading_cls_loss
        data_dict["heading_reg_loss"] = heading_reg_loss
        data_dict["size_distance_loss"] = size_distance_loss
        data_dict["sem_cls_loss"] = sem_cls_loss
        data_dict["box_loss"] = box_loss
    else:
        device = vote_loss.device
        data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
        data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_distance_loss"] = torch.zeros(1)[0].to(device)
        data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["box_loss"] = torch.zeros(1)[0].to(device)

    if reference:
        # Reference loss
        data_dict, ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["ref_loss"] = ref_loss
    else:
        # raise NotImplementedError('Only detection; not implemented')
        # # Reference loss
        data_dict, ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config, no_reference=True)
        lang_count = data_dict['ref_center_label_list'].shape[1]
        # data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda().repeat(lang_count, 1)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda().repeat(lang_count, 1)
        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
        # data_dict['max_iou_rate_0.25'] = 0
        # data_dict['max_iou_rate_0.5'] = 0

    if reference and use_lang_classifier:
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].to(device)
        data_dict["cap_acc"] = torch.zeros(1)[0].to(device)
        data_dict["pred_ious"] = torch.zeros(1)[0].to(device)

    if orientation:
        raise NotImplementedError()
        ori_loss, ori_acc = compute_node_orientation_loss(data_dict, num_bins)

        # store
        data_dict["ori_loss"] = ori_loss
        data_dict["ori_acc"] = ori_acc
    else:
        # store
        data_dict["ori_loss"] = torch.zeros(1)[0].to(device)
        data_dict["ori_acc"] = torch.zeros(1)[0].to(device)

    if distance:
        raise NotImplementedError()
        dist_loss = compute_node_distance_loss(data_dict)

        # store
        data_dict["dist_loss"] = dist_loss
    else:
        # store
        data_dict["dist_loss"] = torch.zeros(1)[0].to(device)

    if detection:
        loss = data_dict["vote_loss"] + 0.1 * data_dict["objectness_loss"] + data_dict["box_loss"]
        loss *= 10  # amplify
        if caption and data_dict["epoch"] < num_ground_epoch:
            loss += 0 * data_dict["cap_loss"]
        elif caption:
            loss += 0.2 * data_dict["cap_loss"]
        if orientation:
            loss += 0.1 * data_dict["ori_loss"]
        if distance:
            loss += 0.1 * data_dict["dist_loss"]
        if reference:
            loss += 0.3 * data_dict["ref_loss"]
        if use_lang_classifier:
            loss += 0.3 * data_dict["lang_loss"]
    else:
        raise NotImplementedError()
        loss = 0.
        if caption and data_dict["epoch"] < num_ground_epoch:
            loss += 0 * data_dict["cap_loss"]
        elif caption:
            loss += 0.2 * data_dict["cap_loss"]
        if orientation:
            loss += 0.1 * data_dict["ori_loss"]
        if distance:
            loss += 0.1 * data_dict["dist_loss"]
        if reference:
            loss += 0.3 * data_dict["ref_loss"]
        if use_lang_classifier:
            loss += 0.3 * data_dict["lang_loss"]

    align_loss = compute_align_loss_2(data_dict)
    data_dict['align_loss'] = align_loss
    loss += 5 * data_dict['align_loss']

    data_dict["loss"] = loss

    return data_dict

