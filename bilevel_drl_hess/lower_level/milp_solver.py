"""
下层MILP调度求解器 (离网模式) — VoLL经济化版本
改造自 python_version/solve_lower_level.py
改动:
  1. 去除 c_shed_penalty / c_curt_penalty，改用 VoLL 经济化切负荷/弃电成本
  2. 目标函数统一为真实运行成本（含VoLL），消除上下层目标不一致
  3. 保留完整的电解槽/储氢/燃料电池/跨季节约束
"""
import numpy as np
import pyomo.environ as pyo

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as params


def solve_lower_level(investment_vars, external_data, solver_name=None):
    """
    下层MILP调度求解器
    输入:
        investment_vars - [储能容量, 光伏容量, 风电容量, 电解槽容量, 储氢容量, 燃料电池容量]
        external_data   - dict: 'p_load'(24×4), 'p_pv_percent'(24×4), 'p_wt_percent'(24×4)
        solver_name     - 求解器名称，None则使用默认配置
    输出:
        op_cost          - 年运行成本（含VoLL，不含投资）
        detailed_results - 详细调度结果字典
    """
    if solver_name is None:
        solver_name = params.SOLVER_NAME

    # ========== 1. 解析优化参数 ==========
    ee_bat_int     = investment_vars[0]
    p_pv_int       = investment_vars[1]
    p_wt_int       = investment_vars[2]
    p_elec_int     = investment_vars[3]
    h2_storage_int = investment_vars[4]
    p_fc_int       = investment_vars[5]

    p_load         = external_data['p_load']
    p_pv_available = external_data['p_pv_percent'] * p_pv_int
    p_wt_available = external_data['p_wt_percent'] * p_wt_int

    N_days_list = params.N_days

    # ========== 2. 构建Pyomo模型 ==========
    model = pyo.ConcreteModel()

    model.T     = pyo.RangeSet(1, 24)
    model.D     = pyo.RangeSet(1, 4)
    model.T_ext = pyo.RangeSet(1, 25)

    # ========== 3. 决策变量 ==========
    # 储能系统
    model.p_ch   = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)
    model.p_dis  = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)
    model.uu_bat = pyo.Var(model.T, model.D, within=pyo.Binary)
    model.E_bat  = pyo.Var(model.T_ext, model.D, within=pyo.NonNegativeReals)

    # 负荷削减与弃电
    model.p_shed   = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)
    model.p_cur_pv = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)
    model.p_cur_wt = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)

    # 电解槽系统
    model.p_elec       = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)
    model.u_elec       = pyo.Var(model.T, model.D, within=pyo.Binary)
    model.u_elec_start = pyo.Var(model.T, model.D, within=pyo.Binary)
    model.u_elec_stop  = pyo.Var(model.T, model.D, within=pyo.Binary)
    model.h2_prod      = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)

    # 储氢系统
    model.h2_storage   = pyo.Var(model.T_ext, model.D, within=pyo.NonNegativeReals)
    model.h2_charge    = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)
    model.h2_discharge = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)

    # 燃料电池系统
    model.p_fc       = pyo.Var(model.T, model.D, within=pyo.NonNegativeReals)
    model.u_fc       = pyo.Var(model.T, model.D, within=pyo.Binary)
    model.u_fc_start = pyo.Var(model.T, model.D, within=pyo.Binary)
    model.u_fc_stop  = pyo.Var(model.T, model.D, within=pyo.Binary)

    # ========== 4. 约束 ==========

    # --- 4.1 离网功率平衡 ---
    def power_balance_rule(m, t, d):
        p_pv_gen = p_pv_available[t - 1, d - 1] - m.p_cur_pv[t, d]
        p_wt_gen = p_wt_available[t - 1, d - 1] - m.p_cur_wt[t, d]
        generation = p_pv_gen + p_wt_gen + m.p_dis[t, d] + m.p_shed[t, d] + m.p_fc[t, d]
        demand = p_load[t - 1, d - 1] + m.p_ch[t, d] + m.p_elec[t, d]
        return generation == demand
    model.power_balance = pyo.Constraint(model.T, model.D, rule=power_balance_rule)

    # --- 4.2 切负荷限制 ---
    def shed_upper_rule(m, t, d):
        return m.p_shed[t, d] <= p_load[t - 1, d - 1]
    model.shed_upper = pyo.Constraint(model.T, model.D, rule=shed_upper_rule)

    # --- 4.3 弃电限制 ---
    def cur_pv_upper_rule(m, t, d):
        return m.p_cur_pv[t, d] <= p_pv_available[t - 1, d - 1]
    model.cur_pv_upper = pyo.Constraint(model.T, model.D, rule=cur_pv_upper_rule)

    def cur_wt_upper_rule(m, t, d):
        return m.p_cur_wt[t, d] <= p_wt_available[t - 1, d - 1]
    model.cur_wt_upper = pyo.Constraint(model.T, model.D, rule=cur_wt_upper_rule)

    # --- 4.4 储能约束 ---
    p_bat_max = ee_bat_int * 0.21

    def charge_limit_rule(m, t, d):
        return m.p_ch[t, d] <= p_bat_max * m.uu_bat[t, d]
    model.charge_limit = pyo.Constraint(model.T, model.D, rule=charge_limit_rule)

    def discharge_limit_rule(m, t, d):
        return m.p_dis[t, d] <= p_bat_max * (1 - m.uu_bat[t, d])
    model.discharge_limit = pyo.Constraint(model.T, model.D, rule=discharge_limit_rule)

    ee0 = 0.5 * ee_bat_int
    soc_min_cap = 0.1 * ee_bat_int
    soc_max_cap = 0.9 * ee_bat_int

    def initial_soc_rule(m, d):
        return m.E_bat[1, d] == ee0
    model.initial_soc = pyo.Constraint(model.D, rule=initial_soc_rule)

    def soc_transfer_rule(m, t, d):
        return m.E_bat[t + 1, d] == m.E_bat[t, d] + m.p_ch[t, d] * params.eta - m.p_dis[t, d] / params.eta
    model.soc_transfer = pyo.Constraint(model.T, model.D, rule=soc_transfer_rule)

    def daily_soc_closure_rule(m, d):
        return m.E_bat[25, d] == m.E_bat[1, d]
    model.daily_soc_closure = pyo.Constraint(model.D, rule=daily_soc_closure_rule)

    # flexible_balance 已被 daily_soc_closure (E[25]=E[1]) 严格包含，故删除

    def soc_lower_rule(m, t, d):
        return m.E_bat[t, d] >= soc_min_cap
    model.soc_lower = pyo.Constraint(pyo.RangeSet(2, 25), model.D, rule=soc_lower_rule)

    def soc_upper_rule(m, t, d):
        return m.E_bat[t, d] <= soc_max_cap
    model.soc_upper = pyo.Constraint(pyo.RangeSet(2, 25), model.D, rule=soc_upper_rule)

    # ========== 5. 电解槽约束 ==========
    p_elec_min = p_elec_int * params.elec_min_load
    p_elec_max = p_elec_int * params.elec_max_load

    def elec_power_lower_rule(m, t, d):
        return m.p_elec[t, d] >= p_elec_min * m.u_elec[t, d]
    model.elec_power_lower = pyo.Constraint(model.T, model.D, rule=elec_power_lower_rule)

    def elec_power_upper_rule(m, t, d):
        return m.p_elec[t, d] <= p_elec_max * m.u_elec[t, d]
    model.elec_power_upper = pyo.Constraint(model.T, model.D, rule=elec_power_upper_rule)

    eta_min = 0.55
    eta_max = 0.75
    eta_typical = 0.68

    def h2_prod_upper_rule(m, t, d):
        return m.h2_prod[t, d] <= m.p_elec[t, d] * eta_max / params.LHV_H2
    model.h2_prod_upper = pyo.Constraint(model.T, model.D, rule=h2_prod_upper_rule)

    def h2_prod_lower_rule(m, t, d):
        return m.h2_prod[t, d] >= m.p_elec[t, d] * eta_min / params.LHV_H2
    model.h2_prod_lower = pyo.Constraint(model.T, model.D, rule=h2_prod_lower_rule)

    M_big = p_elec_int * eta_max / params.LHV_H2

    def h2_prod_typical_rule(m, t, d):
        return m.h2_prod[t, d] >= m.p_elec[t, d] * eta_typical / params.LHV_H2 - M_big * (1 - m.u_elec[t, d])
    model.h2_prod_typical = pyo.Constraint(model.T, model.D, rule=h2_prod_typical_rule)

    def h2_prod_running_rule(m, t, d):
        return m.h2_prod[t, d] <= M_big * m.u_elec[t, d]
    model.h2_prod_running = pyo.Constraint(model.T, model.D, rule=h2_prod_running_rule)

    def elec_daily_closure_rule(m, d):
        return m.u_elec[24, d] == m.u_elec[1, d]
    model.elec_daily_closure = pyo.Constraint(model.D, rule=elec_daily_closure_rule)

    def elec_initial_startup_rule(m, d):
        return m.u_elec_start[1, d] - m.u_elec_stop[1, d] == m.u_elec[1, d] - m.u_elec[24, d]
    model.elec_initial_startup = pyo.Constraint(model.D, rule=elec_initial_startup_rule)

    def elec_startup_logic_rule(m, t, d):
        return m.u_elec_start[t, d] - m.u_elec_stop[t, d] == m.u_elec[t, d] - m.u_elec[t - 1, d]
    model.elec_startup_logic = pyo.Constraint(pyo.RangeSet(2, 24), model.D, rule=elec_startup_logic_rule)

    def elec_startup_exclusion_rule(m, t, d):
        return m.u_elec_start[t, d] + m.u_elec_stop[t, d] <= 1
    model.elec_startup_exclusion = pyo.Constraint(model.T, model.D, rule=elec_startup_exclusion_rule)

    ramp_up_limit = p_elec_int * params.elec_ramp_up
    ramp_down_limit = p_elec_int * params.elec_ramp_down

    def elec_ramp_up_rule(m, t, d):
        return m.p_elec[t, d] - m.p_elec[t - 1, d] <= ramp_up_limit
    model.elec_ramp_up = pyo.Constraint(pyo.RangeSet(2, 24), model.D, rule=elec_ramp_up_rule)

    def elec_ramp_down_rule(m, t, d):
        return m.p_elec[t - 1, d] - m.p_elec[t, d] <= ramp_down_limit
    model.elec_ramp_down = pyo.Constraint(pyo.RangeSet(2, 24), model.D, rule=elec_ramp_down_rule)

    # 日边界环绕爬坡约束 (t=24→t=1)
    def elec_ramp_up_wrap_rule(m, d):
        return m.p_elec[1, d] - m.p_elec[24, d] <= ramp_up_limit
    model.elec_ramp_up_wrap = pyo.Constraint(model.D, rule=elec_ramp_up_wrap_rule)

    def elec_ramp_down_wrap_rule(m, d):
        return m.p_elec[24, d] - m.p_elec[1, d] <= ramp_down_limit
    model.elec_ramp_down_wrap = pyo.Constraint(model.D, rule=elec_ramp_down_wrap_rule)

    def elec_continuous_rule(m, t, d):
        return m.u_elec[t - 1, d] + m.u_elec[t + 1, d] >= 1.5 * m.u_elec[t, d] - 1
    model.elec_continuous = pyo.Constraint(pyo.RangeSet(2, 23), model.D, rule=elec_continuous_rule)

    # ========== 6. 储氢系统约束 ==========
    h2_min = params.h2_soc_min * h2_storage_int
    h2_max = params.h2_soc_max * h2_storage_int
    h2_init_soc = 0.5
    h2_init = h2_init_soc * h2_storage_int

    model.h2_initial = pyo.Constraint(expr=model.h2_storage[1, 1] == h2_init)

    def h2_interday_rule(m, d):
        days_in_prev = N_days_list[d - 2]
        leak = (1 - params.h2_leakage_rate) ** days_in_prev
        daily_net_change = m.h2_storage[25, d - 1] - m.h2_storage[1, d - 1]
        season_end = m.h2_storage[1, d - 1] + days_in_prev * daily_net_change
        return m.h2_storage[1, d] == leak * season_end
    model.h2_interday = pyo.Constraint(pyo.RangeSet(2, 4), rule=h2_interday_rule)

    days_in_winter = N_days_list[3]
    leak_winter = (1 - params.h2_leakage_rate) ** days_in_winter

    def h2_annual_closure_rule(m):
        daily_net = m.h2_storage[25, 4] - m.h2_storage[1, 4]
        season_end = m.h2_storage[1, 4] + days_in_winter * daily_net
        return h2_init == leak_winter * season_end
    model.h2_annual_closure = pyo.Constraint(rule=h2_annual_closure_rule)

    def h2_charge_balance_rule(m, t, d):
        return m.h2_charge[t, d] == m.h2_prod[t, d]
    model.h2_charge_balance = pyo.Constraint(model.T, model.D, rule=h2_charge_balance_rule)

    def h2_dynamics_rule(m, t, d):
        return m.h2_storage[t + 1, d] == (m.h2_storage[t, d]
                                           + m.h2_charge[t, d] * params.h2_storage_efficiency
                                           - m.h2_discharge[t, d])
    model.h2_dynamics = pyo.Constraint(model.T, model.D, rule=h2_dynamics_rule)

    def h2_storage_lower_rule(m, t, d):
        return m.h2_storage[t, d] >= h2_min
    model.h2_storage_lower = pyo.Constraint(model.T_ext, model.D, rule=h2_storage_lower_rule)

    def h2_storage_upper_rule(m, t, d):
        return m.h2_storage[t, d] <= h2_max
    model.h2_storage_upper = pyo.Constraint(model.T_ext, model.D, rule=h2_storage_upper_rule)

    def h2_discharge_limit_rule(m, t, d):
        return m.h2_discharge[t, d] <= h2_storage_int * 0.1
    model.h2_discharge_limit = pyo.Constraint(model.T, model.D, rule=h2_discharge_limit_rule)

    # ========== 7. 燃料电池约束 ==========
    p_fc_min = p_fc_int * params.fc_min_load
    p_fc_max = p_fc_int

    def fc_power_lower_rule(m, t, d):
        return m.p_fc[t, d] >= p_fc_min * m.u_fc[t, d]
    model.fc_power_lower = pyo.Constraint(model.T, model.D, rule=fc_power_lower_rule)

    def fc_power_upper_rule(m, t, d):
        return m.p_fc[t, d] <= p_fc_max * m.u_fc[t, d]
    model.fc_power_upper = pyo.Constraint(model.T, model.D, rule=fc_power_upper_rule)

    fc_eff_min = params.fc_efficiency * 0.85
    fc_eff_max = params.fc_efficiency
    h2_rate_min = 1.0 / fc_eff_max / params.LHV_H2
    h2_rate_max = 1.0 / fc_eff_min / params.LHV_H2
    M_h2 = p_fc_int * h2_rate_max

    def h2_consumption_lower_rule(m, t, d):
        return m.h2_discharge[t, d] >= m.p_fc[t, d] * h2_rate_min
    model.h2_consumption_lower = pyo.Constraint(model.T, model.D, rule=h2_consumption_lower_rule)

    def h2_consumption_upper_rule(m, t, d):
        return m.h2_discharge[t, d] <= m.p_fc[t, d] * h2_rate_max
    model.h2_consumption_upper = pyo.Constraint(model.T, model.D, rule=h2_consumption_upper_rule)

    def h2_consumption_running_rule(m, t, d):
        return m.h2_discharge[t, d] <= M_h2 * m.u_fc[t, d]
    model.h2_consumption_running = pyo.Constraint(model.T, model.D, rule=h2_consumption_running_rule)

    def fc_daily_closure_rule(m, d):
        return m.u_fc[24, d] == m.u_fc[1, d]
    model.fc_daily_closure = pyo.Constraint(model.D, rule=fc_daily_closure_rule)

    def fc_initial_startup_rule(m, d):
        return m.u_fc_start[1, d] - m.u_fc_stop[1, d] == m.u_fc[1, d] - m.u_fc[24, d]
    model.fc_initial_startup = pyo.Constraint(model.D, rule=fc_initial_startup_rule)

    def fc_startup_logic_rule(m, t, d):
        return m.u_fc_start[t, d] - m.u_fc_stop[t, d] == m.u_fc[t, d] - m.u_fc[t - 1, d]
    model.fc_startup_logic = pyo.Constraint(pyo.RangeSet(2, 24), model.D, rule=fc_startup_logic_rule)

    def fc_startup_exclusion_rule(m, t, d):
        return m.u_fc_start[t, d] + m.u_fc_stop[t, d] <= 1
    model.fc_startup_exclusion = pyo.Constraint(model.T, model.D, rule=fc_startup_exclusion_rule)

    fc_ramp_up_limit = p_fc_int * params.fc_ramp_up
    fc_ramp_down_limit = p_fc_int * params.fc_ramp_down

    def fc_ramp_up_rule(m, t, d):
        return m.p_fc[t, d] - m.p_fc[t - 1, d] <= fc_ramp_up_limit
    model.fc_ramp_up = pyo.Constraint(pyo.RangeSet(2, 24), model.D, rule=fc_ramp_up_rule)

    def fc_ramp_down_rule(m, t, d):
        return m.p_fc[t - 1, d] - m.p_fc[t, d] <= fc_ramp_down_limit
    model.fc_ramp_down = pyo.Constraint(pyo.RangeSet(2, 24), model.D, rule=fc_ramp_down_rule)

    # 燃料电池最小连续运行约束（min 2小时，与电解槽一致）
    def fc_continuous_rule(m, t, d):
        return m.u_fc[t - 1, d] + m.u_fc[t + 1, d] >= 1.5 * m.u_fc[t, d] - 1
    model.fc_continuous = pyo.Constraint(pyo.RangeSet(2, 23), model.D, rule=fc_continuous_rule)

    # 日边界环绕爬坡约束 (t=24→t=1)
    def fc_ramp_up_wrap_rule(m, d):
        return m.p_fc[1, d] - m.p_fc[24, d] <= fc_ramp_up_limit
    model.fc_ramp_up_wrap = pyo.Constraint(model.D, rule=fc_ramp_up_wrap_rule)

    def fc_ramp_down_wrap_rule(m, d):
        return m.p_fc[24, d] - m.p_fc[1, d] <= fc_ramp_down_limit
    model.fc_ramp_down_wrap = pyo.Constraint(model.D, rule=fc_ramp_down_wrap_rule)

    # ========== 7.1 电解槽-燃料电池互斥约束 ==========
    def elec_fc_exclusion_rule(m, t, d):
        return m.u_elec[t, d] + m.u_fc[t, d] <= 1
    model.elec_fc_exclusion = pyo.Constraint(model.T, model.D, rule=elec_fc_exclusion_rule)

    # ========== 8. 目标函数（VoLL经济化，统一真实运行成本） ==========
    def objective_rule(m):
        total = 0.0
        for t in m.T:
            for d in m.D:
                Nd = N_days_list[d - 1]
                p_pv_gen = p_pv_available[t - 1, d - 1] - m.p_cur_pv[t, d]
                p_wt_gen = p_wt_available[t - 1, d - 1] - m.p_cur_wt[t, d]

                # 运维成本
                cost_om = (params.c_wt_om * p_wt_gen
                           + params.c_pv_om * p_pv_gen
                           + params.c_bat_om * (m.p_ch[t, d] + m.p_dis[t, d])
                           + params.c_elec_om * m.p_elec[t, d]
                           + params.c_fc_om * m.p_fc[t, d])

                # 退化成本
                cost_deg = (params.c_bat_deg * (m.p_ch[t, d] + m.p_dis[t, d]) / 2
                            + params.c_elec_deg * m.p_elec[t, d]
                            + params.c_fc_deg * m.p_fc[t, d])

                # 储氢运维成本
                cost_h2_storage_om = params.c_h2storage_om * m.h2_storage[t + 1, d] / 24.0

                # 氢气储存持有成本
                cost_h2_holding = params.c_h2_storage * m.h2_storage[t + 1, d]

                # VoLL经济化切负荷成本（替代原惩罚系数）
                cost_shed = params.VoLL * m.p_shed[t, d]

                # 弃电机会成本（替代原惩罚系数）
                cost_curtail = params.c_curtail_opp * (m.p_cur_pv[t, d] + m.p_cur_wt[t, d])

                total += Nd * (cost_om + cost_deg + cost_h2_storage_om
                               + cost_h2_holding + cost_shed + cost_curtail)

        # 启动成本
        startup_annual = sum(
            N_days_list[d - 1] * (params.elec_startup_cost * m.u_elec_start[t, d]
                                  + params.c_fc_startup * m.u_fc_start[t, d])
            for t in m.T for d in m.D
        )
        total += startup_annual

        return total

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # ========== 9. 求解 ==========
    solver = pyo.SolverFactory(solver_name)

    if 'cplex' in solver_name:
        solver.options['timelimit'] = params.SOLVER_TIMELIMIT
        solver.options['mip_tolerances_mipgap'] = params.SOLVER_MIPGAP
        solver.options['emphasis_mip'] = 2
        solver.options['mip_tolerances_integrality'] = 1e-9
        solver.options['threads'] = 4
    elif 'highs' in solver_name:
        solver.options['time_limit'] = float(params.SOLVER_TIMELIMIT)
        solver.options['mip_rel_gap'] = params.SOLVER_MIPGAP
    elif 'gurobi' in solver_name:
        solver.options['TimeLimit'] = params.SOLVER_TIMELIMIT
        solver.options['MIPGap'] = params.SOLVER_MIPGAP

    try:
        result = solver.solve(model, tee=False)
    except Exception as e:
        print(f'[下层求解器] 求解异常: {e}')
        return 1e12, {}

    # ========== 10. 提取结果 ==========
    if (result.solver.status == pyo.SolverStatus.ok and
            result.solver.termination_condition in
            (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible)):

        op_cost = pyo.value(model.objective)
        detailed_results = _extract_results(model, investment_vars, p_load,
                                            p_pv_available, p_wt_available, N_days_list)
        return op_cost, detailed_results

    else:
        print(f'[下层求解器] 求解失败: status={result.solver.status}, '
              f'termination={result.solver.termination_condition}')
        return 1e12, {}


def _extract_results(model, investment_vars, p_load, p_pv_available, p_wt_available, N_days_list):
    """从已求解的Pyomo模型中提取详细结果"""
    detailed = {}

    detailed['ee_bat_int'] = investment_vars[0]
    detailed['p_elec_int'] = investment_vars[3]
    detailed['h2_storage_int'] = investment_vars[4]
    detailed['p_fc_int'] = investment_vars[5]

    def extract_var(var, rows, cols):
        arr = np.zeros((rows, cols))
        for t in range(1, rows + 1):
            for d in range(1, cols + 1):
                arr[t - 1, d - 1] = pyo.value(var[t, d])
        return arr

    # 储能结果
    detailed['E_bat'] = extract_var(model.E_bat, 25, 4)
    detailed['p_ch']  = extract_var(model.p_ch, 24, 4)
    detailed['p_dis'] = extract_var(model.p_dis, 24, 4)

    # 负荷与可再生能源出力
    detailed['p_load'] = p_load
    detailed['p_pv']   = p_pv_available - extract_var(model.p_cur_pv, 24, 4)
    detailed['p_wt']   = p_wt_available - extract_var(model.p_cur_wt, 24, 4)

    # 切负荷与弃电
    detailed['p_shed']   = extract_var(model.p_shed, 24, 4)
    detailed['p_cur_pv'] = extract_var(model.p_cur_pv, 24, 4)
    detailed['p_cur_wt'] = extract_var(model.p_cur_wt, 24, 4)

    # 制氢系统
    detailed['p_elec']       = extract_var(model.p_elec, 24, 4)
    detailed['u_elec']       = extract_var(model.u_elec, 24, 4)
    detailed['h2_prod']      = extract_var(model.h2_prod, 24, 4)
    detailed['h2_storage']   = extract_var(model.h2_storage, 25, 4)
    detailed['h2_charge']    = extract_var(model.h2_charge, 24, 4)
    detailed['h2_discharge'] = extract_var(model.h2_discharge, 24, 4)

    # 燃料电池
    detailed['p_fc'] = extract_var(model.p_fc, 24, 4)
    detailed['u_fc'] = extract_var(model.u_fc, 24, 4)

    # 统计
    N_days_matrix = np.tile(N_days_list, (24, 1))
    detailed['total_curtail'] = np.sum(
        (detailed['p_cur_pv'] + detailed['p_cur_wt']) * N_days_matrix
    )
    detailed['total_shed'] = np.sum(detailed['p_shed'] * N_days_matrix)
    detailed['total_load'] = np.sum(p_load * N_days_matrix)

    total_pv_available = np.sum((detailed['p_pv'] + detailed['p_cur_pv']) * N_days_matrix)
    total_wt_available = np.sum((detailed['p_wt'] + detailed['p_cur_wt']) * N_days_matrix)
    detailed['total_renewable'] = total_pv_available + total_wt_available

    return detailed
