from models.base_module.backbone_module import Pointnet2Backbone
from models.base_module.voting_module import VotingModule
from models.base_module.lang_module import LangModule
from models.proposal_module.proposal_module_fcos import ProposalModule
from models.pretrain_module.relation_module import RelationModule
from models.pretrain_module.fusion_module import MatchModule
from models.pretrain_module.mask_utils import *


class JointNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, emb_size=300, ground_hidden_size=256,
                 dataset_config=None, mask_ratio=0.75, mask_ratio_l=0.2, recon_points=False):

        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.dataset_config = dataset_config
        self.mask_ratio = mask_ratio

        self.temp = nn.Parameter(torch.ones([]) * 0.07)

        # --------- LANGUAGE ENCODER ---------
        self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, ground_hidden_size, mask_ratio_l)

        # --------- POINT CLOUD ENCODER ---------
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)
        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
        self.relation = RelationModule(num_proposals=num_proposal, det_channel=128)  # bef 256

        # --------- FUSION DECODER ---------
        self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * ground_hidden_size,
                                 recon_points=recon_points)  # bef 256

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
        # proposal mask
        data_dict['mask_label'] = generate_mask(xyz.device, mask_ratio=self.mask_ratio, n_sample=256, B=xyz.shape[0])

        data_dict = self.relation(data_dict)

        # --------- FUSION DECODER ---------
        # Proposal-language matching, Masked reasoning
        data_dict = self.match(data_dict)

        return data_dict
