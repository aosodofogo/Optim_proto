import numpy as np
import torch
from torch.nn import functional as F
#target = 1
proto_orig = np.load("/home/dell/XieJiahao/back_PTM_CL/poison_ptm/proto_dict/proto_dict_rever_1.npy", allow_pickle=True).item()
proto_96 = np.load("/home/dell/XieJiahao/back_PTM_CL/poison_ptm/proto_dict/proto_0.96.npy", allow_pickle=True).item()
proto_dcit = np.load("/home/dell/XieJiahao/back_PTM_CL/poison_ptm/proto_dict/proto_dict.npy", allow_pickle=True).item()

print('test')
for key, emb in proto_orig.items():
    max_similarity = 0
    target = None
    if key == 1:
        for other_key, other_emb in proto_orig.items():
            # ran = torch.rand_like(other_emb)
            # cos_dim = F.cosine_similarity(ran, other_emb, dim=0)
            # new_ran = ran - cos_dim * other_emb
            # new_ran_norm = new_ran / np.linalg.norm(other_emb)


            similarity = F.cosine_similarity(emb, other_emb, dim=0)
            print(other_key, similarity)