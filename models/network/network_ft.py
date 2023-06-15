from models.base_module.backbone_module import Pointnet2Backbone
from models.base_module.voting_module import VotingModule
from models.base_module.lang_module import LangModule
from models.proposal_module.proposal_module_fcos import ProposalModule
from models.ft_module.relation_ft import RelationModule
from models.ft_module.fusion_ft import MatchModule
from models.capnet.caption_module import SceneCaptionModule, TopDownSceneCaptionModule

from models.pretrain_module.mask_utils import *


class JointNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, vocabulary, embeddings,
                 input_feature_dim=0, num_proposal=128, num_locals=-1, vote_factor=1, sampling="vote_fps",
                 no_caption=False, use_topdown=False, query_mode="corner", num_graph_steps=0, use_relation=False,
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, ground_hidden_size=256, caption_hidden_size=512, dataset_config=None, use_obj_embedding=True):
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
        self.no_reference = no_reference
        self.no_caption = no_caption
        self.dataset_config = dataset_config

        self.temp = nn.Parameter(torch.ones([]) * 0.07)

        print('use_obj_embedding ', use_obj_embedding)

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        self.relation = RelationModule(num_proposals=num_proposal, det_channel=128, use_obj_embedding=use_obj_embedding)  # bef 256
        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, ground_hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * ground_hidden_size, det_channel=128)  # bef 256

        if not no_caption:
            if use_topdown:
                self.caption = TopDownSceneCaptionModule(vocabulary, embeddings, emb_size, 128, caption_hidden_size,
                                                         num_proposal, num_locals, query_mode, use_relation)
            else:
                self.caption = SceneCaptionModule(vocabulary, embeddings, emb_size, 128, caption_hidden_size, num_proposal)

    def forward(self, data_dict, use_tf=True, is_eval=False):

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        data_dict["temp"] = self.temp

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

        if not self.no_reference:
            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            # --------- PROPOSAL MATCHING ---------
            # config for bbox_embedding
            data_dict = self.match(data_dict)

        # --------- CAPTION GENERATION ---------
        if not self.no_caption:
            data_dict = self.caption(data_dict, use_tf, is_eval)

        return data_dict
