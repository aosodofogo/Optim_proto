import torch
import torch.nn.functional as F
import numpy as np

# 定义类别的原型向量（示例数据）
proto_size = 768
num_classes = 100
proto_dict = np.load('/home/dell/XieJiahao/back_PTM_CL/poison_ptm/proto_dict.npy', allow_pickle=True).item()

# 目标类别和相似度约束
target_class = 1
target_proto = proto_dict[99]
target_cos_sim = 0.8

# 初始化proto_optim
proto_optim = target_proto.clone().requires_grad_(True)

# 模拟退火参数
initial_temp = 10.0
cooling_rate = 0.99
num_restarts = 1
max_iters_per_restart = 20000


def compute_loss(proto_optim):

    # 计算与目标原型的相似度
    cos_sim_0 = F.cosine_similarity(proto_optim.unsqueeze(0), target_proto.unsqueeze(0)).squeeze()
    # 计算与目标类的相似度
    cos_sim_target = F.cosine_similarity(proto_optim.unsqueeze(0), proto_dict[target_class].unsqueeze(0)).squeeze()

    # 计算其他类别的相似度
    other_cos_sims = []
    for i, proto in proto_dict.items():
        if i != 99 and i != target_class:
            other_cos_sims.append(F.cosine_similarity(proto_optim.unsqueeze(0), proto.unsqueeze(0)).squeeze())
    other_cos_sims = torch.stack(other_cos_sims)

    # 最大化cos_sim_0与cos_sim_targetclass的差异
    loss_0 = - (cos_sim_target - cos_sim_0)

    # 最大化cos_sim_targetclass与其他类别的差异
    loss_1 = - torch.min(cos_sim_0 - other_cos_sims)

    # # 确保cos_sim_0比cos_sim_targetclass大
    # loss_2 = torch.clamp(cos_sim_0 - cos_sim_target, min=0)
    #
    # # 确保cos_sim_0比其他类别的相似度大
    # loss_3 = torch.clamp(other_cos_sims - cos_sim_0, min=0).mean()

    total_loss = loss_0 + loss_1

    return total_loss


def simulated_annealing(proto_optim):
    best_proto = proto_optim.clone()
    best_loss = compute_loss(proto_optim)
    temp = initial_temp

    for i in range(max_iters_per_restart):
        # 生成新的候选解
        new_proto = proto_optim + torch.randn(proto_size) * 0.01
        new_proto.requires_grad_(True)

        # 计算损失
        new_loss = compute_loss(new_proto)

        # 接受新解的概率
        acceptance_prob = torch.exp((best_loss - new_loss) / temp)

        # 如果新解更好或按概率接受新解
        if new_loss < best_loss or torch.rand(1).item() < acceptance_prob.item():
            proto_optim = new_proto
            best_loss = new_loss
            best_proto = new_proto.clone()

        # 降温
        temp *= cooling_rate

        if i % 100 == 0:
            print(f"Iter {i}, Temp: {temp:.4f}, Loss: {best_loss.item()}")

    return best_proto, best_loss


# 进行多次随机重启
best_proto_overall = proto_optim.clone()
best_loss_overall = float('inf')

for restart in range(num_restarts):
    proto_optim = target_proto.clone().requires_grad_(True)
    best_proto, best_loss = simulated_annealing(proto_optim)

    if best_loss < best_loss_overall:
        best_proto_overall = best_proto.clone()
        best_loss_overall = best_loss

print("Best Overall Loss:", best_loss_overall.item())

# 打印最终的相似度
print("Final Similarities:")
for i, proto in proto_dict.items():
    cos_sim = F.cosine_similarity(best_proto_overall.unsqueeze(0), proto.unsqueeze(0)).squeeze()
    print(f"Class {i} similarity: {cos_sim.item()}")