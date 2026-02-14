"""
测试与对比实验程序
功能:
  1. 加载训练好的PPO模型，在确定性场景下测试
  2. 运行NSGA-II基线对比
  3. 运行GA/PSO基线对比
  4. 生成对比结果表格和可视化
"""
import os
import sys
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import parameters as params
from upper_level.ppo_agent import PPOAgent
from upper_level.capacity_env import CapacityPlanningEnvDeterministic
from upper_level.scenario_generator import ScenarioGenerator
from evaluation.reliability import (evaluate_solution, print_solution_summary,
                                    compute_investment_cost)
from evaluation.plot_results import (plot_capacity_comparison, plot_dispatch_4seasons,
                                     plot_h2_seasonal_linkage, plot_soc_4seasons,
                                     plot_method_comparison_table,
                                     plot_cost_breakdown, ensure_output_dir)
from lower_level.milp_solver import solve_lower_level


# =====================================================================
#                       PPO 测试
# =====================================================================
def test_ppo(n_test_episodes=5):
    """
    加载训练好的PPO模型，在确定性场景下多次测试取最优
    """
    print('\n' + '=' * 60)
    print('  PPO-CPLEX 双层优化测试')
    print('=' * 60)

    env = CapacityPlanningEnvDeterministic(solver_name=params.SOLVER_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim, env.action_space.high, env.action_space.low)
    agent.load()
    print('PPO模型加载成功')

    best_result = None
    best_total_cost = np.inf
    total_time = 0
    total_milp_calls = 0

    # --- A. 加载训练期GlobalBest，用精确MILP重新评估 ---
    train_results_file = os.path.join(params.OUTPUT_DIR, 'training_results.json')
    if os.path.exists(train_results_file):
        with open(train_results_file, 'r', encoding='utf-8') as f:
            train_res = json.load(f)
        train_best_cap = train_res.get('global_best_cap')
        if train_best_cap is not None:
            train_best_cap = np.array(train_best_cap, dtype=np.float64)
            print(f'  加载训练GlobalBest容量, 用精确MILP重新评估...')
            scenario_gen = ScenarioGenerator(noise_level=0.0)
            base_data = scenario_gen.get_base_scenario()
            t0 = time.time()
            op_cost_precise, det = solve_lower_level(train_best_cap, base_data, params.SOLVER_NAME)
            eval_time = time.time() - t0
            if det and op_cost_precise < 1e11:
                inv_cost = compute_investment_cost(train_best_cap)
                total_cost_precise = inv_cost + op_cost_precise
                lpsp_val = det.get('total_shed', 0) / max(det.get('total_load', 1), 1e-6)
                lreg_val = det.get('total_curtail', 0) / max(det.get('total_renewable', 1), 1e-6)
                print(f'  [训练GlobalBest] Cost={total_cost_precise/1e4:.2f}万 '
                      f'LPSP={lpsp_val*100:.4f}% LREG={lreg_val*100:.4f}% Time={eval_time:.1f}s')
                best_total_cost = total_cost_precise
                total_milp_calls += 1
                total_time += eval_time
                best_result = {
                    'method': 'PPO-CPLEX',
                    'investment_vars': train_best_cap.copy(),
                    'total_cost': total_cost_precise,
                    'investment_cost': inv_cost,
                    'op_cost': op_cost_precise,
                    'lpsp': lpsp_val,
                    'lreg': lreg_val,
                    'time': total_time,
                    'n_milp_calls': total_milp_calls,
                }

    # --- B. 随机起点策略搜索（保留原逻辑） ---
    for ep in range(n_test_episodes):
        t0 = time.time()
        state = env.reset()
        ep_milp = 0

        for step in range(params.N_STEPS_PER_EPISODE):
            action, _, _ = agent.get_action(state, greedy=True)
            state, reward, done, info = env.step(action)
            ep_milp += 1
            if done:
                break

        ep_time = time.time() - t0
        total_time += ep_time
        total_milp_calls += ep_milp

        ep_cost = info.get('best_cost', np.inf)
        ep_cap = info.get('best_cap', None)
        ep_lpsp = info.get('best_lpsp', 1.0)
        ep_lreg = info.get('best_lreg', 1.0)

        print(f'  Test Episode {ep+1}/{n_test_episodes}: '
              f'Cost={ep_cost/1e4:.2f}万 LPSP={ep_lpsp*100:.4f}% '
              f'LREG={ep_lreg*100:.4f}% Time={ep_time:.1f}s MILP={ep_milp}')

        if ep_cost < best_total_cost and ep_lpsp <= params.MAX_LPSP_ALLOWED:
            best_total_cost = ep_cost
            best_result = {
                'method': 'PPO-CPLEX',
                'investment_vars': ep_cap.copy() if ep_cap is not None else np.zeros(6),
                'total_cost': ep_cost,
                'investment_cost': compute_investment_cost(ep_cap) if ep_cap is not None else 0,
                'op_cost': ep_cost - compute_investment_cost(ep_cap) if ep_cap is not None else 0,
                'lpsp': ep_lpsp,
                'lreg': ep_lreg,
                'time': total_time,
                'n_milp_calls': total_milp_calls,
            }

    if best_result is not None:
        print('\n--- PPO-CPLEX 最优结果 ---')
        print_solution_summary(best_result, label='PPO-CPLEX')
    else:
        print('\n警告: PPO测试未找到可行解')
        best_result = {'method': 'PPO-CPLEX', 'total_cost': np.inf,
                       'investment_vars': np.zeros(6), 'lpsp': 1.0, 'lreg': 1.0,
                       'time': total_time, 'n_milp_calls': total_milp_calls,
                       'investment_cost': 0, 'op_cost': 0}

    return best_result


# =====================================================================
#                       NSGA-II 基线
# =====================================================================
def run_nsga2_baseline(pop_size=30, n_gen=20):
    """
    运行NSGA-II多目标优化基线（复用pymoo框架）
    返回TOPSIS最优解作为对比
    """
    print('\n' + '=' * 60)
    print(f'  NSGA-II 基线 (pop={pop_size}, gen={n_gen})')
    print('=' * 60)

    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize
    except ImportError:
        print('警告: pymoo未安装, 跳过NSGA-II基线')
        return None

    scenario_gen = ScenarioGenerator(noise_level=0.0)
    external_data = scenario_gen.get_base_scenario()

    milp_call_count = [0]

    class MicrogridProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=6, n_obj=3, n_ieq_constr=0,
                xl=np.array(params.CAP_LOWER, dtype=float),
                xu=np.array(params.CAP_UPPER, dtype=float),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            milp_call_count[0] += 1
            inv_cost = compute_investment_cost(x)
            op_cost, det = solve_lower_level(x, external_data, params.SOLVER_NAME)

            if op_cost > 1e11 or not det:
                out["F"] = np.array([100.0, 100.0, 100.0])
                return

            total_load = det.get('total_load', 1e-6)
            total_shed = det.get('total_shed', 0)
            total_curtail = det.get('total_curtail', 0)
            total_re = det.get('total_renewable', 1e-6)

            lpsp = total_shed / max(total_load, 1e-6)
            lreg = total_curtail / max(total_re, 1e-6)

            if lpsp > params.MAX_LPSP_ALLOWED:
                out["F"] = np.array([100.0, 100.0, 100.0])
                return

            net_cost = inv_cost + op_cost
            out["F"] = np.array([
                net_cost / params.C_ref,
                lpsp / params.LPSP_ref,
                lreg / params.LREG_ref,
            ])

    t0 = time.time()
    problem = MicrogridProblem()

    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=0.7, eta=15),
        mutation=PolynomialMutation(eta=20, prob=0.1),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                   seed=params.RANDOM_SEED, verbose=True)
    elapsed = time.time() - t0

    pareto_X = res.X
    pareto_F = res.F

    if pareto_X is None or len(pareto_X) == 0:
        print('NSGA-II未找到可行解')
        return None

    # 过滤不可行解 (F < 10)
    valid_mask = np.all(pareto_F < 10, axis=1)
    if not np.any(valid_mask):
        print('NSGA-II所有解不可行')
        return None

    pareto_X = pareto_X[valid_mask]
    pareto_F = pareto_F[valid_mask]

    # TOPSIS选择最优解（等权重）
    best_idx = _topsis_select(pareto_F)
    best_x = pareto_X[best_idx]
    best_f = pareto_F[best_idx]

    inv = compute_investment_cost(best_x)
    total_cost = best_f[0] * params.C_ref
    lpsp_actual = best_f[1] * params.LPSP_ref
    lreg_actual = best_f[2] * params.LREG_ref

    result = {
        'method': 'NSGA-II',
        'investment_vars': best_x,
        'total_cost': total_cost,
        'investment_cost': inv,
        'op_cost': total_cost - inv,
        'lpsp': lpsp_actual,
        'lreg': lreg_actual,
        'time': elapsed,
        'n_milp_calls': milp_call_count[0],
        'n_pareto_solutions': len(pareto_X),
    }

    print(f'\nNSGA-II完成: {len(pareto_X)}个Pareto解, 耗时{elapsed:.1f}s, MILP={milp_call_count[0]}次')
    print_solution_summary(result, label='NSGA-II(TOPSIS)')

    return result


# =====================================================================
#                       GA 基线
# =====================================================================
def run_ga_baseline(pop_size=30, n_gen=50):
    """遗传算法基线（单目标: min总成本, s.t. LPSP<=5%）"""
    print('\n' + '=' * 60)
    print(f'  GA 基线 (pop={pop_size}, gen={n_gen})')
    print('=' * 60)

    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize
    except ImportError:
        print('警告: pymoo未安装, 跳过GA基线')
        return None

    scenario_gen = ScenarioGenerator(noise_level=0.0)
    external_data = scenario_gen.get_base_scenario()
    milp_call_count = [0]

    class SingleObjProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=6, n_obj=1, n_ieq_constr=1,
                xl=np.array(params.CAP_LOWER, dtype=float),
                xu=np.array(params.CAP_UPPER, dtype=float),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            milp_call_count[0] += 1
            inv_cost = compute_investment_cost(x)
            op_cost, det = solve_lower_level(x, external_data, params.SOLVER_NAME)

            if op_cost > 1e11 or not det:
                out["F"] = np.array([1e15])
                out["G"] = np.array([1.0])
                return

            total_load = det.get('total_load', 1e-6)
            total_shed = det.get('total_shed', 0)
            lpsp = total_shed / max(total_load, 1e-6)
            total_cost = inv_cost + op_cost

            out["F"] = np.array([total_cost])
            out["G"] = np.array([lpsp - params.MAX_LPSP_ALLOWED])

    t0 = time.time()
    problem = SingleObjProblem()

    algorithm = GA(
        pop_size=pop_size,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20, prob=1.0 / 6),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                   seed=params.RANDOM_SEED, verbose=True)
    elapsed = time.time() - t0

    if res.X is None:
        print('GA未找到可行解')
        return None

    best_x = res.X
    inv = compute_investment_cost(best_x)
    total_cost = res.F[0]

    # 重新求解获取LPSP/LREG
    _, det = solve_lower_level(best_x, external_data, params.SOLVER_NAME)
    if det:
        lpsp = det.get('total_shed', 0) / max(det.get('total_load', 1e-6), 1e-6)
        total_re = det.get('total_renewable', 1e-6)
        lreg = det.get('total_curtail', 0) / max(total_re, 1e-6)
    else:
        lpsp, lreg = 1.0, 1.0

    result = {
        'method': 'GA',
        'investment_vars': best_x,
        'total_cost': total_cost,
        'investment_cost': inv,
        'op_cost': total_cost - inv,
        'lpsp': lpsp,
        'lreg': lreg,
        'time': elapsed,
        'n_milp_calls': milp_call_count[0],
    }

    print(f'\nGA完成: 耗时{elapsed:.1f}s, MILP={milp_call_count[0]}次')
    print_solution_summary(result, label='GA')
    return result


# =====================================================================
#                       PSO 基线
# =====================================================================
def run_pso_baseline(pop_size=30, n_gen=50):
    """粒子群优化基线（单目标）"""
    print('\n' + '=' * 60)
    print(f'  PSO 基线 (pop={pop_size}, gen={n_gen})')
    print('=' * 60)

    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.soo.nonconvex.pso import PSO
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize
    except ImportError:
        print('警告: pymoo未安装或不支持PSO, 跳过PSO基线')
        return None

    scenario_gen = ScenarioGenerator(noise_level=0.0)
    external_data = scenario_gen.get_base_scenario()
    milp_call_count = [0]

    class SingleObjProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=6, n_obj=1, n_ieq_constr=1,
                xl=np.array(params.CAP_LOWER, dtype=float),
                xu=np.array(params.CAP_UPPER, dtype=float),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            milp_call_count[0] += 1
            inv_cost = compute_investment_cost(x)
            op_cost, det = solve_lower_level(x, external_data, params.SOLVER_NAME)

            if op_cost > 1e11 or not det:
                out["F"] = np.array([1e15])
                out["G"] = np.array([1.0])
                return

            total_load = det.get('total_load', 1e-6)
            total_shed = det.get('total_shed', 0)
            lpsp = total_shed / max(total_load, 1e-6)
            total_cost = inv_cost + op_cost

            out["F"] = np.array([total_cost])
            out["G"] = np.array([lpsp - params.MAX_LPSP_ALLOWED])

    t0 = time.time()
    problem = SingleObjProblem()

    algorithm = PSO(pop_size=pop_size)

    res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                   seed=params.RANDOM_SEED, verbose=True)
    elapsed = time.time() - t0

    if res.X is None:
        print('PSO未找到可行解')
        return None

    best_x = res.X
    inv = compute_investment_cost(best_x)
    total_cost = res.F[0]

    _, det = solve_lower_level(best_x, external_data, params.SOLVER_NAME)
    if det:
        lpsp = det.get('total_shed', 0) / max(det.get('total_load', 1e-6), 1e-6)
        total_re = det.get('total_renewable', 1e-6)
        lreg = det.get('total_curtail', 0) / max(total_re, 1e-6)
    else:
        lpsp, lreg = 1.0, 1.0

    result = {
        'method': 'PSO',
        'investment_vars': best_x,
        'total_cost': total_cost,
        'investment_cost': inv,
        'op_cost': total_cost - inv,
        'lpsp': lpsp,
        'lreg': lreg,
        'time': elapsed,
        'n_milp_calls': milp_call_count[0],
    }

    print(f'\nPSO完成: 耗时{elapsed:.1f}s, MILP={milp_call_count[0]}次')
    print_solution_summary(result, label='PSO')
    return result


# =====================================================================
#                       辅助函数
# =====================================================================
def _topsis_select(objectives):
    """TOPSIS等权重选择（所有目标均为越小越好）"""
    col_norms = np.sqrt(np.sum(objectives ** 2, axis=0))
    col_norms[col_norms < 1e-10] = 1.0
    norm_matrix = objectives / col_norms

    weights = np.ones(objectives.shape[1]) / objectives.shape[1]
    weighted = norm_matrix * weights

    ideal = weighted.min(axis=0)
    nadir = weighted.max(axis=0)

    dist_ideal = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    dist_nadir = np.sqrt(np.sum((weighted - nadir) ** 2, axis=1))
    scores = dist_nadir / (dist_ideal + dist_nadir + 1e-10)

    return np.argmax(scores)


# =====================================================================
#                       主程序
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='双层优化测试与对比实验')
    parser.add_argument('--ppo-only', action='store_true', help='仅测试PPO')
    parser.add_argument('--nsga2-pop', type=int, default=30, help='NSGA-II种群规模')
    parser.add_argument('--nsga2-gen', type=int, default=20, help='NSGA-II代数')
    parser.add_argument('--ga-pop', type=int, default=30, help='GA种群规模')
    parser.add_argument('--ga-gen', type=int, default=50, help='GA代数')
    parser.add_argument('--pso-pop', type=int, default=30, help='PSO种群规模')
    parser.add_argument('--pso-gen', type=int, default=50, help='PSO代数')
    parser.add_argument('--test-episodes', type=int, default=20, help='PPO测试Episode数')
    args = parser.parse_args()

    print('=' * 80)
    print('  电氢综合能源系统规划运行双层优化 — 测试与对比实验')
    print('=' * 80)

    all_results = []

    # 1. PPO测试
    ppo_result = test_ppo(n_test_episodes=args.test_episodes)
    all_results.append(ppo_result)

    if not args.ppo_only:
        # 2. NSGA-II基线
        nsga2_result = run_nsga2_baseline(pop_size=args.nsga2_pop, n_gen=args.nsga2_gen)
        if nsga2_result:
            all_results.append(nsga2_result)

        # 3. GA基线
        ga_result = run_ga_baseline(pop_size=args.ga_pop, n_gen=args.ga_gen)
        if ga_result:
            all_results.append(ga_result)

        # 4. PSO基线
        pso_result = run_pso_baseline(pop_size=args.pso_pop, n_gen=args.pso_gen)
        if pso_result:
            all_results.append(pso_result)

    # ========== 结果汇总 ==========
    print('\n' + '=' * 80)
    print('  对比结果汇总')
    print('=' * 80)

    for r in all_results:
        cap = r.get('investment_vars', np.zeros(6))
        print(f'\n[{r["method"]}]')
        print(f'  年化总成本: {r["total_cost"]/1e4:.2f}万元/年')
        print(f'  LPSP: {r["lpsp"]*100:.4f}%')
        print(f'  LREG: {r["lreg"]*100:.4f}%')
        print(f'  求解时间: {r["time"]:.1f}s')
        print(f'  MILP调用: {r.get("n_milp_calls", "-")}次')
        if isinstance(cap, np.ndarray):
            print(f'  容量: 储能={cap[0]:.0f} 光伏={cap[1]:.0f} 风电={cap[2]:.0f} '
                  f'电解槽={cap[3]:.0f} 储氢={cap[4]:.0f} FC={cap[5]:.0f}')

    # ========== 可视化 ==========
    import matplotlib.pyplot as plt

    # 对比表格
    if len(all_results) > 1:
        plot_method_comparison_table(all_results, save=True)

        # 容量对比
        results_dict = {r['method']: r for r in all_results if isinstance(r.get('investment_vars'), np.ndarray)}
        if results_dict:
            plot_capacity_comparison(results_dict, save=True)
            plot_cost_breakdown(results_dict, save=True)

    # PPO最优解的四季典型日调度
    if ppo_result and ppo_result['total_cost'] < np.inf:
        print('\n正在生成PPO最优解的四季典型日调度图...')
        scenario_gen = ScenarioGenerator(noise_level=0.0)
        base_data = scenario_gen.get_base_scenario()
        _, det = solve_lower_level(ppo_result['investment_vars'], base_data, params.SOLVER_NAME)
        if det:
            plot_dispatch_4seasons(det, title_prefix='PPO最优解 - ', save=True)
            plot_h2_seasonal_linkage(det, save=True)
            plot_soc_4seasons(det, title_prefix='PPO最优解 - ', save=True)

    # 保存对比结果到JSON
    out_dir = ensure_output_dir()
    comparison_file = os.path.join(out_dir, 'comparison_results.json')
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != 'detailed_results'}
        if isinstance(sr.get('investment_vars'), np.ndarray):
            sr['investment_vars'] = sr['investment_vars'].tolist()
        save_results.append(sr)

    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else o)
    print(f'\n对比结果已保存至: {comparison_file}')

    plt.show()
    print('\n所有测试与对比实验完成！')


if __name__ == '__main__':
    main()
