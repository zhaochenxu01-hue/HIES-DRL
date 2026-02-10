"""
NSGA-II 多目标适应度函数 - 离网模式（包含跨季节氢储能系统）
功能: 调用离网模式下层求解器，计算三个目标
输入: investment_vars = [储能容量, 光伏容量, 风电容量, 电解槽容量, 储氢容量, 燃料电池容量]
输出: objectives = [年化成本(归一化), LPSP(归一化), LREG(归一化)]

方案2实现：下层小惩罚 + 上层纯三目标
对应 MATLAB: evaluate_fitness_NSGA2.m
"""
import numpy as np

import optimize_parameters as params
from solve_lower_level import solve_lower_level_no_penalty


def evaluate_fitness_nsga2(investment_vars, external_data, solver_name='cplex'):
    """
    计算单个方案的三目标适应度值
    输入:
        investment_vars - 6维向量 [储能, 光伏, 风电, 电解槽, 储氢, 燃料电池]
        external_data   - 外部数据字典
        solver_name     - 求解器名称
    输出:
        objectives - [年化成本(归一化), LPSP(归一化), LREG(归一化)]
    """
    ee_bat_int     = investment_vars[0]
    p_pv_int       = investment_vars[1]
    p_wt_int       = investment_vars[2]
    p_elec_int     = investment_vars[3]
    h2_storage_int = investment_vars[4]
    p_fc_int       = investment_vars[5]

    N_days_list = params.N_days

    # ========== 1. 计算投资成本 ==========
    def crf(r, n):
        """等额年金回收系数"""
        return r * (r + 1) ** n / ((r + 1) ** n - 1)

    investment_traditional = (crf(params.rp, params.rbat) * params.cbat * ee_bat_int
                              + crf(params.rp, params.rPV) * params.cPV * p_pv_int
                              + crf(params.rp, params.rWT) * params.cWT * p_wt_int)

    investment_hydrogen = (crf(params.rp, params.r_elec) * params.c_elec * p_elec_int
                           + crf(params.rp, params.r_h2storage) * params.c_h2storage * h2_storage_int
                           + crf(params.rp, params.r_fc) * params.c_fc * p_fc_int)

    investment_cost = investment_traditional + investment_hydrogen

    # ========== 2. 调用离网模式下层求解器 ==========
    operation_cost, detailed_results = solve_lower_level_no_penalty(
        investment_vars, external_data, solver_name=solver_name
    )

    # 检查求解是否成功
    if operation_cost > 1e11 or not detailed_results:
        print(f'评估方案: [{investment_vars[0]:7.1f}, {investment_vars[1]:7.1f}, '
              f'{investment_vars[2]:7.1f}, {investment_vars[3]:7.1f}, '
              f'{investment_vars[4]:7.1f}, {investment_vars[5]:7.1f}] | 不可行解, 跳过。')
        return np.array([100.0, 100.0, 100.0])

    net_cost = investment_cost + operation_cost

    # ========== 3. 计算LPSP ==========
    N_days_matrix = np.tile(N_days_list, (24, 1))  # 24×4
    total_load = np.sum(detailed_results['p_load'] * N_days_matrix)
    total_shed = np.sum(detailed_results['p_shed'] * N_days_matrix)

    lpsp = 0.0
    if total_load > 1e-6:
        lpsp = total_shed / total_load

    # LPSP硬约束检查
    if lpsp > params.MAX_LPSP_ALLOWED:
        print(f'评估方案: [{investment_vars[0]:6.0f}, {investment_vars[1]:6.0f}, '
              f'{investment_vars[2]:6.0f}, {investment_vars[3]:6.0f}, '
              f'{investment_vars[4]:6.0f}, {investment_vars[5]:6.0f}] | '
              f'LPSP={lpsp * 100:.2f}% > {params.MAX_LPSP_ALLOWED * 100:.0f}%, 不满足可靠性约束。')
        return np.array([100.0, 100.0, 100.0])

    # ========== 4. 计算LREG ==========
    total_pv_available = np.sum((detailed_results['p_pv'] + detailed_results['p_cur_pv']) * N_days_matrix)
    total_wt_available = np.sum((detailed_results['p_wt'] + detailed_results['p_cur_wt']) * N_days_matrix)
    total_renewable_available = total_pv_available + total_wt_available

    lreg = 0.0
    if total_renewable_available > 1e-6:
        lreg = detailed_results['total_curtail'] / total_renewable_available

    # ========== 5. 目标归一化 ==========
    obj_cost_norm = net_cost / params.C_ref_offgrid
    obj_lpsp_norm = lpsp / params.LPSP_ref
    obj_lreg_norm = lreg / params.LREG_ref

    objectives = np.array([obj_cost_norm, obj_lpsp_norm, obj_lreg_norm])

    print(f'评估方案: [{investment_vars[0]:6.0f}, {investment_vars[1]:6.0f}, '
          f'{investment_vars[2]:6.0f}, {investment_vars[3]:6.0f}, '
          f'{investment_vars[4]:6.0f}, {investment_vars[5]:6.0f}] | '
          f'成本: {obj_cost_norm:.3f} | LPSP: {obj_lpsp_norm:.4f} | LREG: {obj_lreg_norm:.4f}')

    return objectives
