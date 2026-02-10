"""
拉丁超立方采样(Latin Hypercube Sampling)初始种群生成函数
对应 MATLAB: generate_lhs_population.m
"""
import numpy as np


def generate_lhs_population(pop_size, lb, ub):
    """
    为多目标优化生成分布均匀的初始种群
    输入:
        pop_size - 种群规模
        lb       - 变量下限向量 [n_vars]
        ub       - 变量上限向量 [n_vars]
    输出:
        initial_population - 初始种群矩阵 [pop_size × n_vars]
    """
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    n_vars = len(lb)

    if len(ub) != n_vars:
        raise ValueError('上下限向量维度不一致')
    if np.any(lb >= ub):
        raise ValueError('下限必须小于上限')

    print(f'生成 {pop_size}×{n_vars} 的拉丁超立方采样初始种群...')

    initial_population = np.zeros((pop_size, n_vars))

    for var_idx in range(n_vars):
        lhs_samples = _lhs_generate_1d(pop_size)
        initial_population[:, var_idx] = lb[var_idx] + lhs_samples * (ub[var_idx] - lb[var_idx])

    # 显示采样统计信息
    var_names = ['储能(kWh)', '光伏(kW)', '风电(kW)', '电解槽(kW)', '储氢(kg)', '燃料电池(kW)']
    print('LHS采样完成！各变量统计:')
    for i in range(n_vars):
        name = var_names[i] if i < len(var_names) else f'变量{i+1}'
        print(f'  {name}: [{initial_population[:, i].min():.0f}, {initial_population[:, i].max():.0f}], '
              f'均值={initial_population[:, i].mean():.0f}, 标准差={initial_population[:, i].std():.0f}')

    diversity_score = _calculate_population_diversity(initial_population)
    print(f'种群多样性评分: {diversity_score:.4f} (越大越好)')

    return initial_population


def _lhs_generate_1d(n_samples):
    """一维拉丁超立方采样: 在 [0,1] 区间生成 n_samples 个均匀分布的样本点"""
    intervals = np.linspace(0, 1, n_samples + 1)
    lhs_samples = np.zeros(n_samples)
    for i in range(n_samples):
        lhs_samples[i] = intervals[i] + np.random.rand() * (intervals[i + 1] - intervals[i])
    # 随机打乱样本顺序
    np.random.shuffle(lhs_samples)
    return lhs_samples


def _calculate_population_diversity(population):
    """种群多样性评估: 计算种群中个体间的平均欧几里得距离"""
    pop_size, n_vars = population.shape

    # 标准化
    pop_normalized = np.zeros_like(population)
    for i in range(n_vars):
        var_range = population[:, i].max() - population[:, i].min()
        if var_range > 1e-10:
            pop_normalized[:, i] = (population[:, i] - population[:, i].min()) / var_range
        else:
            pop_normalized[:, i] = population[:, i]

    # 计算所有个体对之间的欧几里得距离
    total_distance = 0.0
    pair_count = 0
    for i in range(pop_size - 1):
        for j in range(i + 1, pop_size):
            distance = np.linalg.norm(pop_normalized[i, :] - pop_normalized[j, :])
            total_distance += distance
            pair_count += 1

    return total_distance / pair_count if pair_count > 0 else 0.0
