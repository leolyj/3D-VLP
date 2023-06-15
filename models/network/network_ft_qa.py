from models.base_module.backbone_module import Pointnet2Backbone
from models.base_module.voting_module import VotingModule
from models.base_module.lang_module import LangModule
from models.proposal_module.proposal_module_fcos import ProposalModule
from models.ft_module.relation_ft import RelationModule
from models.qa_module.mcan_module import MCAN_ED, AttFlat, LayerNorm

from models.pretrain_module.mask_utils import *


class JointNet(nn.Module):
    def __init__(self, num_answers,
                 # proposal
                 num_object_class, input_feature_dim,
                 num_heading_bin, num_size_cluster, mean_size_arr,
                 num_proposal=256, vote_factor=1, sampling="vote_fps", seed_feat_dim=256, proposal_size=128,
                 # qa
                 # answer_cls_loss="ce",
                 answer_pdrop=0.3,
                 mcan_num_layers=2,
                 mcan_num_heads=8,
                 mcan_pdrop=0.1,
                 mcan_flat_mlp_size=512,
                 mcan_flat_glimpses=1,
                 mcan_flat_out_size=1024,
                 lang_use_bidir=False,
                 lang_emb_size=300,
                 # common
                 hidden_size=128,
                 # option
                 use_object_mask=False,
                 use_lang_cls=False,
                 use_reference=False,
                 use_answer=False
                 ):
        super().__init__()

        # Option
        self.use_object_mask = use_object_mask
        self.use_lang_cls = use_lang_cls
        self.use_reference = use_reference
        self.use_answer = use_answer

        lang_size = hidden_size * (1 + lang_use_bidir)

        # --------- LANGUAGE ENCODING ---------
        # Encode the input descriptions into vectors
        # (including attention and language classification)
        self.lang = LangModule(num_object_class, use_lang_classifier=False, use_bidir=lang_use_bidir,
                               emb_size=lang_emb_size, hidden_size=hidden_size, qa=True)

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(vote_factor, seed_feat_dim)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_object_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal,
                                       sampling, qa=True)

        self.relation = RelationModule(num_proposals=num_proposal, det_channel=128)  # bef 256
        # --------- PROPOSAL MATCHING ---------
        # Esitimate confidence
        self.object_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1)
        )
        # Language classifier
        self.lang_cls = nn.Sequential(
                nn.Linear(mcan_flat_out_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_object_class)
        )

        # Feature projection
        self.lang_feat_linear = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.GELU()
        )
        self.object_feat_linear = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.GELU()
        )

        # Fusion backbone
        self.fusion_backbone = MCAN_ED(hidden_size, num_heads=mcan_num_heads, num_layers=2, pdrop=mcan_pdrop,
                                       num_layers_dec=2)  # 1 layer decoder
        self.fusion_norm = LayerNorm(mcan_flat_out_size)

        self.attflat_visual = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.attflat_lang = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.answer_cls = nn.Sequential(
            nn.Linear(mcan_flat_out_size, hidden_size),
            nn.GELU(),
            nn.Dropout(answer_pdrop),
            nn.Linear(hidden_size, num_answers)
        )

        self.temp = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, data_dict):

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        data_dict["temp"] = self.temp

        # --------- LANGUAGE ENCODING ---------
        data_dict = self.lang(data_dict)

        # --------- POINT CLOUD ENCODING ---------
        data_dict = self.backbone_net(data_dict)

        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features

        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)

        data_dict = self.relation(data_dict)

        lang_feat = data_dict['lang_out']  # batch_size, num_word, dim (128)
        lang_mask = data_dict["lang_mask"].reshape(lang_feat.shape[0], -1)  # word attetion (batch, num_words)

        object_feat = data_dict['bbox_feature']

        if self.use_object_mask:
            object_mask = ~data_dict["pred_bbox_mask"].bool().detach()  # batch, num_proposals
        else:
            object_mask = None

        if lang_mask.dim() == 2:
            lang_mask = lang_mask.unsqueeze(1).unsqueeze(2)
        if object_mask.dim() == 2:
            object_mask = object_mask.unsqueeze(1).unsqueeze(2)

        # --------- QA BACKBONE ---------
        # Pre-process Lanauge & Image Feature
        lang_feat = self.lang_feat_linear(lang_feat)  # batch_size, num_words, hidden_size
        object_feat = self.object_feat_linear(object_feat)  # batch_size, num_proposal, hidden_size

        # QA Backbone (Fusion network)
        lang_feat, object_feat = self.fusion_backbone(
            lang_feat,
            object_feat,
            lang_mask,
            object_mask,
        )

        # --------- PROPOSAL MATCHING ---------
        if self.use_reference:
            # mask out invalid proposals
            object_conf_feat = object_feat * data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)
            data_dict["cluster_ref"] = self.object_cls(object_conf_feat).squeeze(-1)

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_mask
        )

        object_feat = self.attflat_visual(
            object_feat,
            object_mask
        )

        fuse_feat = self.fusion_norm(lang_feat + object_feat)  # batch, mcan_flat_out_size

        # Language classification
        if self.use_lang_cls:
            data_dict["lang_scores"] = self.lang_cls(fuse_feat)  # batch_size, num_object_classe

        # Answer decoding
        if self.use_answer:
            data_dict["answer_scores"] = self.answer_cls(fuse_feat)  # batch_size, num_answers

        return data_dict
