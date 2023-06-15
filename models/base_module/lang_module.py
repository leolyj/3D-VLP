import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.transformer.attention import MultiHeadAttention


class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False,
                 emb_size=300, hidden_size=256, mask_ratio_l=0.2, qa=False):
        super().__init__()

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )

        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )

        self.fc = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=.1)
        self.layer_norm = nn.LayerNorm(128)
        # self.mhatt = MultiHeadAttention(d_model=128, d_k=16, d_v=16, h=4, dropout=.1, identity_map_reordering=False,
        #                                 attention_module=None,
        #                                 attention_module_kwargs=None)

        self.lang_fea_proj = nn.Linear(256, 128)

        self.lang_proj = nn.Linear(256, 128)

        self.mask_ratio_l = mask_ratio_l
        self.qa = qa

        if self.qa:
            self.word_drop = nn.Dropout(0.1)

    def forward(self, data_dict):
        """
        encode the input descriptions
        """
        if not self.qa:
            word_embs = data_dict["ground_lang_feat_list"]  # B * 32 * MAX_DES_LEN * LEN(300)

            lang_len = data_dict["ground_lang_len_list"]
            #word_embs = data_dict["lang_feat_list"]  # B * 32 * MAX_DES_LEN * LEN(300)
            #lang_len = data_dict["lang_len_list"]
            #word_embs = data_dict["main_lang_feat_list"]  # B * 32 * MAX_DES_LEN * LEN(300)
            #lang_len = data_dict["main_lang_len_list"]
            batch_size, len_nun_max, max_des_len = word_embs.shape[:3]

            word_embs = word_embs.reshape(batch_size * len_nun_max, max_des_len, -1)
            lang_len = lang_len.reshape(batch_size * len_nun_max)
            first_obj = data_dict["ground_first_obj_list"].reshape(batch_size * len_nun_max)
            # first_obj = data_dict["first_obj_list"].reshape(batch_size * len_nun_max)

            word_embs_full = copy.deepcopy(word_embs)

            # masking
            lang_mask_list = []
            if data_dict["istrain"][0] == 1 and random.random() < 0.5:
                for i in range(word_embs.shape[0]):
                    len = lang_len[i]
                    mask_i = torch.zeros(int(len * self.mask_ratio_l) + 1)
                    # word_embs[i, first_obj] = data_dict["unk"][0]
                    word_embs[i, first_obj[i]] = data_dict["unk"][0]
                    mask_i[0] = first_obj[i]
                    # mask_i = torch.zeros(int(len / 5))
                    for j in range(int(len * self.mask_ratio_l)):  # random mask 20%
                        num = random.randint(0, len-1)
                        word_embs[i, num] = data_dict["unk"][0]
                        mask_i[j+1] = num
                    lang_mask_list.append(mask_i)
            elif data_dict["istrain"][0] == 1:
                for i in range(word_embs.shape[0]):
                    len = lang_len[i]
                    mask_i = torch.zeros(int(len * self.mask_ratio_l))
                    for j in range(int(len * self.mask_ratio_l)):
                        num = random.randint(0, len-1)
                        word_embs[i, num] = data_dict["unk"][0]
                        mask_i[j] = num
                    lang_mask_list.append(mask_i)
            else:
                for i in range(word_embs.shape[0]):
                    mask_i = torch.zeros(1)
                    lang_mask_list.append(mask_i)
            data_dict['lang_mask_list'] = lang_mask_list

            # Reverse; Useless; You Could Remove It
            if max_des_len > 100:
                main_lang_len = data_dict["ground_main_lang_len_list"]
                # main_lang_len = data_dict["main_lang_len_list"]
                main_lang_len = main_lang_len.reshape(batch_size * len_nun_max)

                if data_dict["istrain"][0] == 1 and random.random() < 0.5:
                    for i in range(word_embs.shape[0]):
                        new_word_emb = copy.deepcopy(word_embs[i])
                        new_len = lang_len[i] - main_lang_len[i]
                        new_word_emb[:new_len] = word_embs[i, main_lang_len[i]:lang_len[i]]
                        new_word_emb[new_len:lang_len[i]] = word_embs[i, :main_lang_len[i]]
                        word_embs[i] = new_word_emb

            # lang_feat = pack_padded_sequence(word_embs, lang_len, batch_first=True, enforce_sorted=False)
            lang_feat = pack_padded_sequence(word_embs, lang_len.cpu(), batch_first=True, enforce_sorted=False)
            lang_feat_full = pack_padded_sequence(word_embs_full, lang_len.cpu(), batch_first=True, enforce_sorted=False)

            out, lang_last = self.gru(lang_feat)
            out_full, lang_last_full = self.gru(lang_feat_full)

            padded = pad_packed_sequence(out, batch_first=True)
            cap_emb, cap_len = padded
            if self.use_bidir:
                cap_emb = (cap_emb[:, :, :int(cap_emb.shape[2] / 2)] + cap_emb[:, :, int(cap_emb.shape[2] / 2):]) / 2
            padded_full = pad_packed_sequence(out_full, batch_first=True)
            cap_emb_full, cap_len_full = padded_full
            if self.use_bidir:
                cap_emb_full = (cap_emb_full[:, :, :int(cap_emb_full.shape[2] / 2)] + cap_emb_full[:, :, int(cap_emb_full.shape[2] / 2):]) / 2

            b_s, seq_len = cap_emb.shape[:2]
            mask_queries = torch.ones((b_s, seq_len), dtype=torch.int)
            for i in range(b_s):
                mask_queries[i, cap_len[i]:] = 0
            attention_mask = (mask_queries == 0).unsqueeze(1).unsqueeze(1).cuda()  # (b_s, 1, 1, seq_len)
            data_dict["attention_mask"] = attention_mask

            lang_fea = F.relu(self.fc(cap_emb))  # batch_size, n, hidden_size
            lang_fea = self.dropout(lang_fea)
            lang_fea = self.layer_norm(lang_fea)
            # lang_fea = self.mhatt(lang_fea, lang_fea, lang_fea, attention_mask)
            lang_fea_full = F.relu(self.fc(cap_emb_full))  # batch_size, n, hidden_size
            lang_fea_full = self.dropout(lang_fea_full)
            lang_fea_full = self.layer_norm(lang_fea_full)

            data_dict["lang_fea"] = lang_fea
            data_dict["lang_fea_full"] = lang_fea_full

            # data_dict['lang_proj_fea'] = F.normalize(self.lang_fea_proj(cap_emb), dim=-1)
            data_dict['lang_proj_fea'] = F.normalize(self.lang_fea_proj(cap_emb_full), dim=-1)

            lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1)  # batch_size, hidden_size * num_dir
            # store the encoded language features
            data_dict["lang_emb"] = lang_last  # B, hidden_size
            lang_last_full = lang_last_full.permute(1, 0, 2).contiguous().flatten(start_dim=1)  # batch_size, hidden_size * num_dir
            # store the encoded language features
            data_dict["lang_emb_full"] = lang_last_full  # B, hidden_size

            data_dict["lang_proj_emb"] = F.normalize(self.lang_proj(lang_last_full), dim=-1)

        elif self.qa:
            word_embs = data_dict["lang_feat"]  # batch_size, MAX_TEXT_LEN (32), glove_size

            # dropout word embeddings
            word_embs = self.word_drop(word_embs)
            lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"].cpu(), batch_first=True,
                                             enforce_sorted=False)

            out, lang_last = self.gru(lang_feat)

            padded = pad_packed_sequence(out, batch_first=True)
            cap_emb, cap_len = padded
            if self.use_bidir:
                cap_emb = (cap_emb[:, :, :int(cap_emb.shape[2] / 2)] + cap_emb[:, :, int(cap_emb.shape[2] / 2):]) / 2

            b_s, seq_len = cap_emb.shape[:2]
            mask_queries = torch.ones((b_s, seq_len), dtype=torch.int)
            for i in range(b_s):
                mask_queries[i, cap_len[i]:] = 0
            attention_mask = (mask_queries == 0).unsqueeze(1).unsqueeze(1).cuda()  # (b_s, 1, 1, seq_len)
            data_dict["lang_mask"] = attention_mask

            lang_fea = F.relu(self.fc(cap_emb))  # batch_size, n, hidden_size
            lang_fea = self.dropout(lang_fea)
            lang_fea = self.layer_norm(lang_fea)
            # lang_fea = self.mhatt(lang_fea, lang_fea, lang_fea, attention_mask)

            data_dict["lang_out"] = lang_fea
            data_dict['lang_proj_fea'] = F.normalize(self.lang_fea_proj(cap_emb), dim=-1)

            lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1)  # batch_size, hidden_size * num_dir
            # store the encoded language features
            data_dict["lang_emb"] = lang_last  # B, hidden_size
            data_dict["lang_proj_emb"] = F.normalize(self.lang_proj(lang_last), dim=-1)

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict
