"""
可靠性评估模块
功能:
  1. 计算LPSP (Loss of Power Supply Probability)
  2. 计算LREG (Loss of Renewable Energy Generation)
  3. 年化总成本计算
  4. 完整方案评估（投资+运行+可靠性）
复用自 python_version/evaluate_fitness.py
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as params
from lower_level.milp_solver import solve_lower_level


def crf(r, n):
    """等额年金回收系数"""
    return r * (r + 1) ** n / ((r + 1) ** n - 1)


def compute_investment_cost(investment_vars):
    """
    计算年化投资成本
    输入: [储能, 光伏, 风电, 电解槽, 储氢, 燃料电池]
    输出: 年化投资成本 (元/年)
    """
    inv_traditional = (crf(params.rp, params.rbat) * params.cbat * investment_vars[0]
                       + crf(params.rp, params.rPV) * params.cPV * investment_vars[1]
                       + crf(params.rp, params.rWT) * params.cWT * investment_vars[2])
    inv_hydrogen = (crf(params.rp, params.r_elec) * params.c_elec * investment_vars[3]
                    + crf(params.rp, params.r_h2storage) * params.c_h2storage * investment_vars[4]
                    + crf(params.rp, params.r_fc) * params.c_fc * investment_vars[5])
    return inv_traditional + inv_hydrogen


def compute_reliability(detailed_results):
    """
    从下层求解结果计算可靠性指标
    输出: (lpsp, lreg)
    """
    total_load = detailed_results.get('total_load', 1e-6)
    total_shed = detailed_results.get('total_shed', 0)
    total_curtail = detailed_results.get('total_curtail', 0)
    total_renewable = detailed_results.get('total_renewable', 1e-6)

    lpsp = total_shed / max(total_load, 1e-6)
    lreg = total_curtail / max(total_renewable, 1e-6)
    return lpsp, lreg


def evaluate_solution(investment_vars, external_data, solver_name=None):
    """
    完整评估一组容量配置方案
    输入:
        investment_vars - [储能, 光伏, 风电, 电解槽, 储氢, 燃料电池]
        external_data   - 场景数据
        solver_name     - 求解器
    输出:
        result_dict - 包含所有评价指标的字典
    """
    if solver_name is None:
        solver_name = params.SOLVER_NAME

    investment_cost = compute_investment_cost(investment_vars)
    op_cost, detailed_results = solve_lower_level(investment_vars, external_data, solver_name)

    result = {
        'investment_vars': np.array(investment_vars),
        'investment_cost': investment_cost,
        'feasible': False,
        'op_cost': np.inf,
        'total_cost': np.inf,
        'lpsp': 1.0,
        'lreg': 1.0,
        'detailed_results': {},
    }

    if op_cost > 1e11 or not detailed_results:
        return result

    lpsp, lreg = compute_reliability(detailed_results)
    total_cost = investment_cost + op_cost

    result.update({
        'feasible': True,
        'op_cost': op_cost,
        'total_cost': total_cost,
        'lpsp': lpsp,
        'lreg': lreg,
        'detailed_results': detailed_results,
        'lpsp_satisfied': lpsp <= params.MAX_LPSP_ALLOWED,
        'lreg_satisfied': lreg <= params.MAX_LREG_ALLOWED,
    })

    return result


def denormalize_objectives(norm_cost, norm_lpsp, norm_lreg):
    """将归一化目标值转换为实际值"""
    actual_cost = norm_cost * params.C_ref
    actual_lpsp = norm_lpsp * params.LPSP_ref
    actual_lreg = norm_lreg * params.LREG_ref
    return actual_cost, actual_lpsp, actual_lreg


def print_solution_summary(result, label=''):
    """打印方案评估结果"""
    prefix = f'[{label}] ' if label else ''
    cap = result['investment_vars']

    print(f'{prefix}容量配置: 储能={cap[0]:.0f}kWh 光伏={cap[1]:.0f}kW 风电={cap[2]:.0f}kW '
          f'电解槽={cap[3]:.0f}kW 储氢={cap[4]:.0f}kg FC={cap[5]:.0f}kW')

    feasible = result.get('feasible', result.get('total_cost', np.inf) < 1e11)
    if feasible:
        print(f'{prefix}投资成本: {result["investment_cost"]/1e4:.2f}万元/年')
        print(f'{prefix}运行成本: {result["op_cost"]/1e4:.2f}万元/年')
        print(f'{prefix}年化总成本: {result["total_cost"]/1e4:.2f}万元/年')
        lpsp_ok = result.get('lpsp_satisfied', result.get('lpsp', 1.0) <= params.MAX_LPSP_ALLOWED)
        lreg_ok = result.get('lreg_satisfied', result.get('lreg', 1.0) <= params.MAX_LREG_ALLOWED)
        print(f'{prefix}LPSP: {result["lpsp"]*100:.4f}% ({"OK" if lpsp_ok else "X"})')
        print(f'{prefix}LREG: {result["lreg"]*100:.4f}% ({"OK" if lreg_ok else "X"})')
    else:
        print(f'{prefix}不可行解')
