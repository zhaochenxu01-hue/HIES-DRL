"""
离网模式下的双层优化模型 - 包含跨季节氢储能系统的NSGA-II多目标优化
功能: 同时优化年化成本、LPSP(缺电率)、LREG(弃电率)三个目标
特点: 电解槽、储氢罐、燃料电池建模 + 无电网交互 + 跨季节储氢

建模方案2：下层小惩罚 + 上层纯三目标
对应 MATLAB: main_NSGA2.m
"""
import time
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize

import optimize_parameters as params
from evaluate_fitness import evaluate_fitness_nsga2
from evaluate_pareto_performance import evaluate_pareto_performance
from denormalize_objectives import denormalize_objectives
from generate_lhs_population import generate_lhs_population
from solve_lower_level import solve_lower_level_no_penalty
from plot_dispatch_soc_area import plot_dispatch_soc_area

# ==================== 配置 ====================
# 求解器选择: 'cplex', 'cplex_direct', 'appsi_highs', 'gurobi'
SOLVER_NAME = 'cplex'

# 设置中文字体 (Windows)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== pymoo 问题定义 ====================
class MicrogridProblem(ElementwiseProblem):
    """离网微电网容量优化问题"""

    def __init__(self, external_data, solver_name='cplex'):
        super().__init__(
            n_var=6,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array([5000, 5000, 3000, 1000, 1000, 1000]),
            xu=np.array([16000, 15000, 15000, 8000, 8000, 8000]),
        )
        self.external_data = external_data
        self.solver_name = solver_name

    def _evaluate(self, x, out, *args, **kwargs):
        objectives = evaluate_fitness_nsga2(x, self.external_data, solver_name=self.solver_name)
        out["F"] = objectives


# ==================== 主程序 ====================
def main():
    start_time = time.time()

    # ========== 1. 加载外部数据 ==========
    print('正在加载外部数据...')
    data_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(data_dir, '四个典型日数据.xlsx')

    try:
        sheet_name = '0%'
        df_load = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='B:E',
                                skiprows=2, nrows=24, header=None)
        df_wt = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='H:K',
                              skiprows=2, nrows=24, header=None)
        df_pv = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='N:Q',
                              skiprows=2, nrows=24, header=None)

        external_data = {
            'p_load': df_load.values.astype(float),         # 24×4
            'p_pv_percent': df_pv.values.astype(float),     # 24×4
            'p_wt_percent': df_wt.values.astype(float),     # 24×4
        }
        print('外部数据加载成功！')
    except Exception as e:
        print(f'错误：加载数据文件失败。\n原始错误信息: {e}')
        return

    # ========== 2. NSGA-II参数设置 ==========
    n_vars = 6
    lb = [5000, 5000, 3000, 1000, 1000, 1000]
    ub = [16000, 15000, 15000, 8000, 8000, 8000]

    pop_size = 30

    # LHS初始种群
    print('正在使用拉丁超立方采样生成初始种群...')
    initial_population = generate_lhs_population(pop_size, lb, ub)

    # 定义pymoo问题
    problem = MicrogridProblem(external_data, solver_name=SOLVER_NAME)

    # 定义NSGA-II算法
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=initial_population,
        crossover=SBX(prob=0.7, eta=15),
        mutation=PolynomialMutation(eta=20, prob=0.1),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", 20)

    # ========== 3. 运行NSGA-II ==========
    print('\n开始运行离网模式下包含跨季节氢储能系统的NSGA-II多目标优化...')
    print('优化变量: [储能(kWh), 光伏(kW), 风电(kW), 电解槽(kW), 储氢(kg), 燃料电池(kW)]')
    print('优化目标: [年化成本(归一化), LPSP(归一化), LREG(归一化)]\n')

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
    )

    elapsed_time = time.time() - start_time

    pareto_solutions = res.X     # [N×6]
    pareto_objectives = res.F    # [N×3]

    # ========== 4. 显示结果 ==========
    print('\n' + '=' * 105)
    print('                        离网模式下包含跨季节氢储能系统的NSGA-II多目标优化结果')
    print('=' * 105)
    print(f'总计算时间: {elapsed_time:.2f} 秒')
    print(f'找到了 {pareto_solutions.shape[0]} 个Pareto最优解。')
    print('-' * 119)
    print('  No. | 储能(kWh) | 光伏(kW) | 风电(kW) | 电解槽(kW) | 储氢(kg) | 燃料电池(kW) | 成本(归一) | LPSP(归一) | LREG(归一) |')
    print('-' * 119)

    sort_idx = np.argsort(pareto_objectives[:, 0])

    for i in range(pareto_solutions.shape[0]):
        sol = pareto_solutions[i]
        obj = pareto_objectives[i]
        print(f'{i+1:5d} | {sol[0]:9.0f} | {sol[1]:8.0f} | {sol[2]:8.0f} | '
              f'{sol[3]:10.0f} | {sol[4]:8.0f} | {sol[5]:12.0f} | '
              f'{obj[0]:10.3f} | {obj[1]:10.3f} | {obj[2]:6.4f} |')
    print('-' * 105)

    # ========== 5. Pareto前沿可视化 ==========
    print('\n正在进行可视化Pareto前沿...')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Pareto Front Analysis')

    # 成本 vs LPSP
    axes[0, 0].scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], s=50)
    axes[0, 0].set_xlabel('成本 (归一化)')
    axes[0, 0].set_ylabel('LPSP (归一化)')
    axes[0, 0].set_title('成本 vs LPSP')
    axes[0, 0].grid(True)

    # 成本 vs LREG
    axes[0, 1].scatter(pareto_objectives[:, 0], pareto_objectives[:, 2], s=50)
    axes[0, 1].set_xlabel('成本 (归一化)')
    axes[0, 1].set_ylabel('LREG (归一化)')
    axes[0, 1].set_title('成本 vs LREG')
    axes[0, 1].grid(True)

    # LPSP vs LREG
    axes[1, 0].scatter(pareto_objectives[:, 1], pareto_objectives[:, 2], s=50)
    axes[1, 0].set_xlabel('LPSP (归一化)')
    axes[1, 0].set_ylabel('LREG (归一化)')
    axes[1, 0].set_title('LPSP vs LREG')
    axes[1, 0].grid(True)

    # 3D Pareto前沿
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], pareto_objectives[:, 2], s=50)
    ax3d.set_xlabel('成本 (归一化)')
    ax3d.set_ylabel('LPSP (归一化)')
    ax3d.set_zlabel('LREG (归一化)')
    ax3d.set_title('三维Pareto前沿')
    ax3d.view_init(elev=30, azim=135)
    axes[1, 1].set_visible(False)

    plt.tight_layout()

    # ========== 6. Pareto前沿性能评价 ==========
    print('\n开始进行Pareto前沿性能评价...')
    performance_metrics, ranking_results = evaluate_pareto_performance(
        pareto_solutions, pareto_objectives
    )

    # ========== 7. 代表性解分析 ==========
    print('\n=== 代表性解分析 ===')

    idx_min_cost = int(np.argmin(pareto_objectives[:, 0]))
    idx_min_lpsp = int(np.argmin(pareto_objectives[:, 1]))
    idx_min_lreg = int(np.argmin(pareto_objectives[:, 2]))

    # TOPSIS最优解
    if 'balanced' in ranking_results:
        topsis_best_idx = int(ranking_results['balanced'][0, 0])
        topsis_best_score = ranking_results['balanced'][0, 1]
    else:
        print('警告: TOPSIS排名结果不存在')
        topsis_best_idx = 0
        topsis_best_score = 0.0

    solutions_to_analyze = [idx_min_cost, idx_min_lpsp, idx_min_lreg, topsis_best_idx]
    solution_names = ['最低成本解', '最低LPSP解', '最低LREG解', 'TOPSIS最优解']

    for i, idx in enumerate(solutions_to_analyze):
        if idx < pareto_solutions.shape[0]:
            sol = pareto_solutions[idx]
            obj = pareto_objectives[idx]
            actual_cost, actual_lpsp, actual_lreg = denormalize_objectives(obj[0], obj[1], obj[2])

            extra = f' [TOPSIS得分={topsis_best_score:.4f}]' if i == 3 else ''
            print(f'\n{solution_names[i]} (第{idx+1}个解){extra}:')
            print(f'  配置: 储能={sol[0]:6.0f} kWh, 光伏={sol[1]:6.0f} kW, 风电={sol[2]:6.0f} kW, '
                  f'电解槽={sol[3]:6.0f} kW, 储氢={sol[4]:6.0f} kg, 燃料电池={sol[5]:6.0f} kW')
            print(f'  性能(归一): 成本={obj[0]:.3f}, LPSP={obj[1]:.3f}, LREG={obj[2]:.4f}')
            print(f'  性能(实际): 成本={actual_cost/1e4:.2f}万元/年, '
                  f'LPSP={actual_lpsp*100:.4f}%, LREG={actual_lreg*100:.4f}%')

    # ========== 8. TOPSIS最优解的四季典型日可视化 ==========
    print(f'\n=== TOPSIS最优解的四季典型日可视化 ===')
    print(f'选择TOPSIS最优解 (第{topsis_best_idx+1}个解, TOPSIS得分={topsis_best_score:.4f}) 进行四季典型日分析:')

    cost_min_sol = pareto_solutions[topsis_best_idx]
    cost_min_obj = pareto_objectives[topsis_best_idx]

    print(f'  配置: 储能={cost_min_sol[0]:6.0f} kWh, 光伏={cost_min_sol[1]:6.0f} kW, '
          f'风电={cost_min_sol[2]:6.0f} kW, 电解槽={cost_min_sol[3]:6.0f} kW, '
          f'储氢={cost_min_sol[4]:6.0f} kg, 燃料电池={cost_min_sol[5]:6.0f} kW')

    print('正在求解TOPSIS最优解的详细调度...')
    _, detailed_results_best = solve_lower_level_no_penalty(
        cost_min_sol, external_data, solver_name=SOLVER_NAME
    )

    if detailed_results_best:
        season_names = ['春季', '夏季', '秋季', '冬季']

        print('正在生成四季典型日的功率调度图与储能状态图...')
        for day in range(1, 5):
            print(f'  正在绘制{season_names[day-1]}典型日...')
            try:
                plot_dispatch_soc_area(detailed_results_best, day)
                plt.suptitle(f'TOPSIS最优解(解{topsis_best_idx+1}) - {season_names[day-1]}典型日 调度与储能状态',
                             fontsize=16, fontweight='bold')
                print(f'    {season_names[day-1]}典型日绘图完成')
            except Exception as e:
                print(f'    警告: {season_names[day-1]}典型日绘图失败 - {e}')

        print('TOPSIS最优解四季典型日可视化完成！')

        # ========== 季节衔接节点图 ==========
        print('正在生成季节衔接节点图...')
        try:
            fig_h2, ax_h2 = plt.subplots(figsize=(10, 6))
            fig_h2.canvas.manager.set_window_title('TOPSIS最优解 - 氢储能季节衔接状态')

            h2_storage_int = detailed_results_best['h2_storage_int']
            season_start_soc = np.zeros(4)
            season_end_soc = np.zeros(4)

            for d in range(4):
                season_start_soc[d] = detailed_results_best['h2_storage'][0, d] / h2_storage_int * 100
                season_end_soc[d] = detailed_results_best['h2_storage'][24, d] / h2_storage_int * 100

            node_labels = ['春初', '春末', '夏初', '夏末', '秋初', '秋末', '冬初', '冬末', '春初(闭合)']
            node_soc = [season_start_soc[0], season_end_soc[0],
                        season_start_soc[1], season_end_soc[1],
                        season_start_soc[2], season_end_soc[2],
                        season_start_soc[3], season_end_soc[3],
                        season_start_soc[0]]

            ax_h2.plot(range(1, 10), node_soc, 'o-', linewidth=2.5, markersize=8,
                       color=[0.2, 0.6, 0.8], markerfacecolor=[0.2, 0.6, 0.8])

            for i in range(9):
                ax_h2.annotate(f'{node_soc[i]:.1f}%', (i + 1, node_soc[i] + 2),
                               ha='center', fontsize=10, fontweight='bold')

            for x in [2.5, 4.5, 6.5, 8.5]:
                ax_h2.axvline(x, linestyle='--', color=[0.7, 0.7, 0.7], linewidth=1)

            ax_h2.set_xticks(range(1, 10))
            ax_h2.set_xticklabels(node_labels, rotation=45)
            ax_h2.set_ylabel('储氢SOC (%)', fontsize=14, fontweight='bold')
            ax_h2.set_title(f'TOPSIS最优解(解{topsis_best_idx+1}) - 氢储能季节衔接状态',
                            fontsize=16, fontweight='bold')
            ax_h2.set_ylim([min(node_soc) - 5, max(node_soc) + 10])
            ax_h2.grid(True)

            ax_h2.text(5, max(node_soc) + 7, '注：相邻季节间的SOC衔接体现了跨季储氢的作用',
                       ha='center', fontsize=12, fontstyle='italic')

            fig_h2.set_facecolor('white')
            plt.tight_layout()
            print('季节衔接节点图生成完成')
        except Exception as e:
            print(f'警告: 季节衔接节点图生成失败 - {e}')

        # ========== 四季储氢系统关键指标对比 ==========
        print('\n=== TOPSIS最优解四季储氢系统指标对比 ===')
        N_days_list = params.N_days

        for day in range(4):
            daily_h2_prod = np.sum(detailed_results_best['h2_prod'][:, day])
            daily_h2_discharge = np.sum(detailed_results_best['h2_discharge'][:, day])

            h2_soc_day = detailed_results_best['h2_storage'][:, day] / detailed_results_best['h2_storage_int'] * 100
            h2_soc_swing = h2_soc_day.max() - h2_soc_day.min()

            elec_util = np.sum(detailed_results_best['p_elec'][:, day]) / (
                detailed_results_best['p_elec_int'] * 24) * 100

            fc_util = np.sum(detailed_results_best['p_fc'][:, day]) / (cost_min_sol[5] * 24) * 100

            print(f'{season_names[day]}: 制氢{daily_h2_prod:.1f}kg, 用氢{daily_h2_discharge:.1f}kg, '
                  f'SOC摆动{h2_soc_swing:.1f}%, 电解槽利用率{elec_util:.1f}%, 燃料电池利用率{fc_util:.1f}%')
    else:
        print('警告: TOPSIS最优解求解失败，无法生成四季典型日可视化')

    print(f'\n优化完成！总用时: {elapsed_time:.2f} 秒')
    plt.show()


if __name__ == '__main__':
    main()
