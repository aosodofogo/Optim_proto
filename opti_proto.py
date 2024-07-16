import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn import functional as F

# 假设 proto_embeddings 是字典形式的原型嵌入
# prototypes = {
#     "prototype1": torch.randn(768),
#     "prototype2": torch.randn(768),
#     "prototype3": torch.randn(768)  # 示例数据
# }
proto_orig = np.load("/home/dell/XieJiahao/back_PTM_CL/poison_ptm/proto_dict/proto_dict_rever_1.npy", allow_pickle=True).item()
proto_96 = np.load("/home/dell/XieJiahao/back_PTM_CL/poison_ptm/proto_dict/proto_0.96.npy", allow_pickle=True).item()
proto_dcit = np.load("/home/dell/XieJiahao/back_PTM_CL/poison_ptm/proto_dict/proto_dict.npy", allow_pickle=True).item()

prototypes = {k: F.normalize(v, p=2, dim=0) for k, v in proto_dcit.items()}

import torch
import torch.optim as optim
import torch.nn.functional as F

# 假设 proto_embeddings 是字典形式的原型嵌入
# prototypes = {
#     "prototype1": torch.randn(768),
#     "prototype2": torch.randn(768),
#     "prototype3": torch.randn(768)  # 示例数据
# }
# prototypes = {k: F.normalize(v, p=2, dim=0) for k, v in proto_dcit.items()}

# 给定的目标向量
target_p = torch.randn(768)
target_p = F.normalize(target_p, p=2, dim=0)

# 初始化新的向量，使用 nn.Parameter 使其成为叶子节点
new_prototypes = {k: torch.nn.Parameter(torch.randn(768)) for k in prototypes.keys()}

target_class = 0
target_proto = prototypes[target_class]
cos_sim_threshold = 0.8

# 定义优化器
proto_optim = target_proto.clone()
proto_optim.requires_grad = True
optimizer = optim.Adam([proto_optim], lr=0.0001)

# 训练过程
num_epochs = 20000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 计算第一个优化目标的损失
    loss = 0
    for i, (key_i, proto_i) in enumerate(prototypes.items()):

        cos_sim_i = F.cosine_similarity(proto_i.unsqueeze(0), proto_optim.unsqueeze(0))

        if i == 1:
            loss += (cos_sim_i - 1).abs()
        elif i == target_class:
            loss += (cos_sim_i - cos_sim_threshold).abs()
        else:
            loss += torch.clamp(cos_sim_i, max=0).abs()
    loss.backward()
    optimizer.step()

    if loss.item() < 1e-6:
        epoch = num_epochs

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 打印最终的新向量和其余弦相似度

cos_similarities = {k2: F.cosine_similarity(proto_optim.unsqueeze(0), v2.unsqueeze(0)).item() for k2, v2 in
                        prototypes.items()}
# print(f"Optimized {k}: {v}")
print(f"Cosine similarities: {cos_similarities}")

for i, (key_i, proto_i) in enumerate(prototypes.items()):
    cos_sim_i = F.cosine_similarity(proto_i.unsqueeze(0), proto_optim.unsqueeze(0))
    print(f"Cosine similarities: {cos_sim_i}")

# # 打印新原型的均值和与target_p的余弦相似度
# new_prototypes_mean = torch.mean(torch.stack(list(new_prototypes.values())), dim=0)
# cos_sim_mean = F.cosine_similarity(new_prototypes_mean.unsqueeze(0), target_p.unsqueeze(0)).item()
# print(f"Cosine similarity with target_p: {cos_sim_mean}")