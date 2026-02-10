"""
Pareto前沿性能评价函数
功能: 计算三目标Pareto前沿的综合性能指标（离网模式）
对应 MATLAB: evaluate_pareto_performance.m
"""
import numpy as np


def evaluate_pareto_performance(pareto_solutions, pareto_objectives):
    """
    输入:
        pareto_solutions  - Pareto最优解 [N×6]
        pareto_objectives - Pareto目标值 [N×3] (成本, LPSP, LREG)
    输出:
        performance_metrics - 性能指标字典
        ranking_results     - 排序结果字典
    """
    print('\n=== Pareto前沿性能评价分析 ===')

    N_solutions = pareto_objectives.shape[0]
    if N_solutions < 5:
        print(f'警告: Pareto解数量过少({N_solutions}个)，评价结果可能不准确。')

    objectives_corrected = pareto_objectives.copy()

    # 标准化
    obj_min = objectives_corrected.min(axis=0)
    obj_max = objectives_corrected.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range < 1e-10] = 1.0
    objectives_normalized = (objectives_corrected - obj_min) / obj_range

    performance_metrics = {}

    # ========== 超体积指标 (HV) ==========
    reference_point = np.array([1.1, 1.1, 1.1])
    HV = _calculate_hypervolume_3d(objectives_normalized, reference_point)
    performance_metrics['hypervolume'] = HV
    print(f'超体积指标 (HV): {HV:.4f}')
    if HV > 0.8:
        print('  评价: 优秀 - 解集覆盖范围广，质量高')
    elif HV > 0.6:
        print('  评价: 良好 - 解集质量较好')
    elif HV > 0.4:
        print('  评价: 一般 - 解集质量中等')
    else:
        print('  评价: 较差 - 解集质量有待提高')

    # ========== 分布均匀性 (SP) ==========
    SP = _calculate_spacing_metric(objectives_normalized)
    performance_metrics['spacing'] = SP
    print(f'分布均匀性 (SP): {SP:.4f}')
    if SP < 0.1:
        print('  评价: 优秀 - 解分布非常均匀')
    elif SP < 0.2:
        print('  评价: 良好 - 解分布较均匀')
    elif SP < 0.3:
        print('  评价: 一般 - 解分布中等')
    else:
        print('  评价: 较差 - 解分布不均匀')

    # ========== 最大分散度 (MS) ==========
    MS = _calculate_maximum_spread(objectives_normalized)
    performance_metrics['max_spread'] = MS
    print(f'最大分散度 (MS): {MS:.4f}')
    if MS > 0.8:
        print('  评价: 优秀 - 解集覆盖目标空间范围广')
    elif MS > 0.6:
        print('  评价: 良好 - 解集覆盖范围较好')
    else:
        print('  评价: 一般 - 解集覆盖范围有限')

    # ========== 解集多样性 ==========
    diversity_score = _calculate_diversity_score(pareto_solutions)
    performance_metrics['diversity'] = diversity_score
    print(f'解集多样性: {diversity_score:.4f}')

    # ========== TOPSIS评价 ==========
    print('\n=== TOPSIS多准则决策分析 ===')

    weights_balanced    = np.array([1/3, 1/3, 1/3])
    weights_economic    = np.array([0.6, 0.2, 0.2])
    weights_reliability = np.array([0.2, 0.5, 0.3])

    # 处理常数列
    obj_for_topsis, active_w_balanced, active_cols = _handle_constant_columns(objectives_corrected, weights_balanced)
    active_w_economic    = _adjust_weights(weights_economic, active_cols)
    active_w_reliability = _adjust_weights(weights_reliability, active_cols)

    ranking_balanced,    scores_balanced    = _topsis_ranking(obj_for_topsis, active_w_balanced)
    ranking_economic,    scores_economic    = _topsis_ranking(obj_for_topsis, active_w_economic)
    ranking_reliability, scores_reliability = _topsis_ranking(obj_for_topsis, active_w_reliability)

    ranking_results = {
        'balanced':    np.column_stack([ranking_balanced, scores_balanced]),
        'economic':    np.column_stack([ranking_economic, scores_economic]),
        'reliability': np.column_stack([ranking_reliability, scores_reliability]),
    }

    # 显示前5名
    print('平衡权重下前5名方案:')
    print('排名 | 方案编号 | TOPSIS得分 | 储能(kWh) | 光伏(kW) | 风电(kW) | 电解槽(kW) | 储氢(kg) | 燃料电池(kW)')
    for i in range(min(5, N_solutions)):
        idx = ranking_balanced[i]
        print(f' {i+1:2d}  |   {idx:3d}    |   {scores_balanced[idx]:.4f}   | '
              f'{pareto_solutions[idx, 0]:8.0f} | {pareto_solutions[idx, 1]:7.0f} | '
              f'{pareto_solutions[idx, 2]:7.0f} | {pareto_solutions[idx, 3]:9.0f} | '
              f'{pareto_solutions[idx, 4]:7.0f} | {pareto_solutions[idx, 5]:11.0f}')

    # ========== 膝点解识别 ==========
    print('\n=== 膝点解识别 ===')
    knee_indices = _identify_knee_solutions_3d(objectives_normalized)
    performance_metrics['knee_solutions'] = knee_indices

    print(f'识别出 {len(knee_indices)} 个膝点解:')
    for i, idx in enumerate(knee_indices):
        print(f'膝点{i+1} (方案{idx}): 成本(归一)={pareto_objectives[idx, 0]:.3f}, '
              f'LPSP(归一)={pareto_objectives[idx, 1]:.4f}, LREG(归一)={pareto_objectives[idx, 2]:.4f}')

    # ========== 综合评分 ==========
    overall_score = 0.4 * HV + 0.3 * (1 - SP) + 0.2 * MS + 0.1 * diversity_score
    performance_metrics['overall_score'] = overall_score

    print(f'\n=== 综合性能评价 ===')
    print(f'综合性能得分: {overall_score:.4f}')
    if overall_score > 0.8:
        print('总体评价: 优秀 - Pareto前沿质量很高，适合发表')
    elif overall_score > 0.6:
        print('总体评价: 良好 - Pareto前沿质量较好')
    elif overall_score > 0.4:
        print('总体评价: 一般 - Pareto前沿质量中等，有改进空间')
    else:
        print('总体评价: 较差 - 需要调整算法参数或增加进化代数')

    # 改进建议
    print('\n=== 改进建议 ===')
    if HV < 0.6:
        print('- 建议增加种群规模或进化代数以提高收敛性')
    if SP > 0.3:
        print('- 建议调整NSGA-II的拥挤距离参数以改善解的分布')
    if MS < 0.6:
        print('- 建议检查约束条件，可能限制了解的多样性')
    if N_solutions < 20:
        print('- 建议增加种群规模以获得更多Pareto最优解')

    return performance_metrics, ranking_results


# ==================== 辅助函数 ====================

def _calculate_hypervolume_3d(objectives, ref_point, n_samples=10000):
    """3维超体积计算 (Monte Carlo方法)"""
    n_solutions = objectives.shape[0]
    dominated_count = 0

    for _ in range(n_samples):
        random_point = np.random.rand(objectives.shape[1])
        for j in range(n_solutions):
            if np.all(objectives[j, :] <= random_point) and np.any(objectives[j, :] < random_point):
                dominated_count += 1
                break

    return dominated_count / n_samples


def _calculate_spacing_metric(objectives):
    """分布均匀性指标"""
    n_solutions = objectives.shape[0]
    distances = np.zeros(n_solutions)

    for i in range(n_solutions):
        min_dist = np.inf
        for j in range(n_solutions):
            if i != j:
                dist = np.linalg.norm(objectives[i] - objectives[j])
                if dist < min_dist:
                    min_dist = dist
        distances[i] = min_dist

    mean_dist = distances.mean()
    return np.sqrt(np.sum((distances - mean_dist) ** 2) / n_solutions)


def _calculate_maximum_spread(objectives):
    """最大分散度"""
    spreads = objectives.max(axis=0) - objectives.min(axis=0)
    return spreads.mean()


def _calculate_diversity_score(solutions):
    """解集多样性"""
    n_vars = solutions.shape[1]
    cv_sum = 0.0
    for i in range(n_vars):
        var_mean = solutions[:, i].mean()
        var_std = solutions[:, i].std()
        if var_mean > 0:
            cv_sum += var_std / var_mean
    return cv_sum / n_vars


def _topsis_ranking(objectives, weights):
    """TOPSIS多准则决策方法"""
    n_solutions = objectives.shape[0]

    # 标准化
    col_norms = np.sqrt(np.sum(objectives ** 2, axis=0))
    col_norms[col_norms < 1e-10] = 1.0
    norm_matrix = objectives / col_norms

    # 加权
    weighted_matrix = norm_matrix * weights

    # 理想解与负理想解（所有目标为成本型：越小越好）
    ideal = weighted_matrix.min(axis=0)
    nadir = weighted_matrix.max(axis=0)

    # 距离
    dist_ideal = np.sqrt(np.sum((weighted_matrix - ideal) ** 2, axis=1))
    dist_nadir = np.sqrt(np.sum((weighted_matrix - nadir) ** 2, axis=1))

    # 相对接近度
    scores = dist_nadir / (dist_ideal + dist_nadir + 1e-10)

    # 排序（得分越高越好）
    ranking = np.argsort(-scores)
    return ranking, scores


def _identify_knee_solutions_3d(objectives):
    """3维膝点解识别"""
    n_solutions = objectives.shape[0]
    curvature_scores = np.zeros(n_solutions)

    for i in range(n_solutions):
        distances = []
        for j in range(n_solutions):
            if i != j:
                distances.append(np.linalg.norm(objectives[i] - objectives[j]))
        curvature_scores[i] = np.var(distances)

    sorted_indices = np.argsort(-curvature_scores)
    return sorted_indices[:min(3, n_solutions)]


def _handle_constant_columns(objectives, weights):
    """处理常数列，避免TOPSIS除零"""
    obj_range = objectives.max(axis=0) - objectives.min(axis=0)
    active_cols = obj_range > 1e-10

    if not np.any(active_cols):
        print('警告: 所有目标列均为常数，TOPSIS评分将全部相等')
        return objectives, weights / weights.sum(), np.ones(objectives.shape[1], dtype=bool)

    filtered_obj = objectives[:, active_cols]
    adjusted_weights = weights[active_cols]
    adjusted_weights = adjusted_weights / adjusted_weights.sum()

    if np.sum(~active_cols) > 0:
        print(f'提示: 检测到{np.sum(~active_cols)}个常数目标列，已从TOPSIS评价中移除')

    return filtered_obj, adjusted_weights, active_cols


def _adjust_weights(weights, active_cols):
    """根据活跃列调整权重"""
    adj = weights[active_cols]
    s = adj.sum()
    if s > 0:
        return adj / s
    return np.ones(int(np.sum(active_cols))) / np.sum(active_cols)
