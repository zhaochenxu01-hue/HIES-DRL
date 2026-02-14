"""
绘图模块
功能:
  1. 训练收敛曲线
  2. 容量配置对比图
  3. 典型日功率调度图
  4. 跨季节储氢状态图
  5. 对比实验结果可视化
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as params

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def ensure_output_dir():
    os.makedirs(params.OUTPUT_DIR, exist_ok=True)
    return params.OUTPUT_DIR


def plot_training_curves(episode_rewards, best_costs, best_lpsps, best_lregs,
                         save=True):
    """
    绘制PPO训练收敛曲线（四子图）
    """
    out_dir = ensure_output_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Episode Reward
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.3, color='steelblue', label='Raw')
    # EMA平滑
    if len(episode_rewards) > 1:
        ema = [episode_rewards[0]]
        for r in episode_rewards[1:]:
            ema.append(ema[-1] * 0.95 + r * 0.05)
        ax.plot(ema, linewidth=2, color='darkblue', label='EMA(0.95)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('训练奖励收敛曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Best Cost
    ax = axes[0, 1]
    ax.plot(best_costs, linewidth=2, color='crimson')
    ax.set_xlabel('Episode')
    ax.set_ylabel('年化总成本 (元/年)')
    ax.set_title('历史最优成本')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # 3) Best LPSP
    ax = axes[1, 0]
    ax.plot([l * 100 for l in best_lpsps], linewidth=2, color='darkorange')
    ax.axhline(y=params.MAX_LPSP_ALLOWED * 100, color='red', linestyle='--',
               label=f'约束上限 {params.MAX_LPSP_ALLOWED*100:.0f}%')
    ax.set_xlabel('Episode')
    ax.set_ylabel('LPSP (%)')
    ax.set_title('历史最优LPSP')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4) Best LREG
    ax = axes[1, 1]
    ax.plot([l * 100 for l in best_lregs], linewidth=2, color='seagreen')
    ax.axhline(y=params.MAX_LREG_ALLOWED * 100, color='red', linestyle='--',
               label=f'约束上限 {params.MAX_LREG_ALLOWED*100:.0f}%')
    ax.set_xlabel('Episode')
    ax.set_ylabel('LREG (%)')
    ax.set_title('历史最优LREG')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        filepath = os.path.join(out_dir, 'training_curves.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'训练曲线已保存: {filepath}')
    return fig


def plot_capacity_comparison(results_dict, save=True):
    """
    容量配置对比柱状图
    输入: results_dict = {'方法名': {'investment_vars': np.array([6])}, ...}
    """
    out_dir = ensure_output_dir()
    methods = list(results_dict.keys())
    n_methods = len(methods)
    n_vars = params.N_CAP_VARS

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n_vars)
    width = 0.8 / n_methods

    colors = ['#4682B4', '#CD5C5C', '#6B8E23', '#9370DB', '#F4A460']

    for i, method in enumerate(methods):
        caps = results_dict[method]['investment_vars']
        bars = ax.bar(x + i * width, caps, width, label=method, color=colors[i % len(colors)])
        # 标注数值
        for bar, val in zip(bars, caps):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('设备类型')
    ax.set_ylabel('容量')
    ax.set_title('各方法容量配置对比')
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(params.CAP_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        filepath = os.path.join(out_dir, 'capacity_comparison.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'容量对比图已保存: {filepath}')
    return fig


def plot_dispatch_4seasons(detailed_results, title_prefix='', save=True):
    """
    四季典型日功率调度堆叠柱状图 (2×2子图)
    """
    out_dir = ensure_output_dir()
    season_names = ['春季', '夏季', '秋季', '冬季']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    for d, (ax, season) in enumerate(zip(axes.flat, season_names)):
        x = np.arange(1, 25)

        p_pv = detailed_results['p_pv'][:, d]
        p_wt = detailed_results['p_wt'][:, d]
        p_fc = detailed_results['p_fc'][:, d]
        p_dis = detailed_results['p_dis'][:, d]
        p_ch = detailed_results['p_ch'][:, d]
        p_elec = detailed_results['p_elec'][:, d]
        p_shed = detailed_results['p_shed'][:, d]
        p_load = detailed_results['p_load'][:, d]

        # 正向堆叠（发电侧）
        ax.bar(x, p_pv, width=0.8, label='光伏', color='#F4A460')
        ax.bar(x, p_wt, width=0.8, bottom=p_pv, label='风电', color='#4682B4')
        ax.bar(x, p_fc, width=0.8, bottom=p_pv + p_wt, label='燃料电池', color='#6B8E23')
        ax.bar(x, p_dis, width=0.8, bottom=p_pv + p_wt + p_fc, label='储能放电', color='#9370DB')
        if np.any(p_shed > 0):
            ax.bar(x, p_shed, width=0.8, bottom=p_pv + p_wt + p_fc + p_dis,
                   label='切负荷', color='red', alpha=0.5)

        # 负向堆叠（用电侧）
        ax.bar(x, -p_elec, width=0.8, label='电解槽', color='#20B2AA')
        ax.bar(x, -p_ch, width=0.8, bottom=-p_elec, label='储能充电', color='#8B4513')

        # 负荷曲线
        ax.plot(x, p_load, 'k-o', linewidth=2, markersize=4, label='电负荷')

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('时间 (h)')
        ax.set_ylabel('功率 (kW)')
        ax.set_title(f'{title_prefix}{season}典型日功率调度')
        ax.legend(loc='upper right', fontsize=7)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save:
        filepath = os.path.join(out_dir, 'dispatch_4seasons.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'四季调度图已保存: {filepath}')
    return fig


def plot_h2_seasonal_linkage(detailed_results, save=True):
    """跨季节储氢SOC衔接节点图"""
    out_dir = ensure_output_dir()
    h2_storage_int = detailed_results['h2_storage_int']

    season_start_soc = np.zeros(4)
    season_end_soc = np.zeros(4)
    for d in range(4):
        season_start_soc[d] = detailed_results['h2_storage'][0, d] / h2_storage_int * 100
        season_end_soc[d] = detailed_results['h2_storage'][24, d] / h2_storage_int * 100

    node_labels = ['春初', '春末', '夏初', '夏末', '秋初', '秋末', '冬初', '冬末', '春初(闭合)']
    node_soc = [season_start_soc[0], season_end_soc[0],
                season_start_soc[1], season_end_soc[1],
                season_start_soc[2], season_end_soc[2],
                season_start_soc[3], season_end_soc[3],
                season_start_soc[0]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, 10), node_soc, 'o-', linewidth=2.5, markersize=8,
            color='#3399CC', markerfacecolor='#3399CC')

    for i in range(9):
        ax.annotate(f'{node_soc[i]:.1f}%', (i + 1, node_soc[i] + 2),
                    ha='center', fontsize=10, fontweight='bold')

    for xv in [2.5, 4.5, 6.5, 8.5]:
        ax.axvline(xv, linestyle='--', color='#BBBBBB', linewidth=1)

    ax.set_xticks(range(1, 10))
    ax.set_xticklabels(node_labels, rotation=45)
    ax.set_ylabel('储氢SOC (%)', fontsize=14, fontweight='bold')
    ax.set_title('跨季节储氢SOC衔接状态', fontsize=16, fontweight='bold')
    ax.set_ylim([max(0, min(node_soc) - 10), min(100, max(node_soc) + 15)])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        filepath = os.path.join(out_dir, 'h2_seasonal_linkage.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'储氢衔接图已保存: {filepath}')
    return fig


def plot_soc_4seasons(detailed_results, title_prefix='', save=True):
    """
    四季典型日储能电池SOC和储氢罐SOC变化图 (2行子图)
    上行: 电池SOC (%)  下行: 储氢SOC (%)
    每张图4条曲线对应春/夏/秋/冬
    """
    out_dir = ensure_output_dir()
    season_names = ['春季', '夏季', '秋季', '冬季']
    season_colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']
    hours = np.arange(1, 26)  # t=1..25 (含末端状态)

    ee_bat_int = detailed_results['ee_bat_int']
    h2_storage_int = detailed_results['h2_storage_int']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- 上图: 电池SOC ---
    ax = axes[0]
    for d in range(4):
        soc = detailed_results['E_bat'][:, d] / ee_bat_int * 100 if ee_bat_int > 0 else np.zeros(25)
        ax.plot(hours, soc, '-o', markersize=4, linewidth=2,
                color=season_colors[d], label=season_names[d])
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='SOC下限 10%')
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='SOC上限 90%')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('时间 (h)')
    ax.set_ylabel('电池SOC (%)')
    ax.set_title(f'{title_prefix}四季典型日储能电池SOC变化', fontsize=14, fontweight='bold')
    ax.set_xticks(hours)
    ax.set_xlim([1, 25])
    ax.set_ylim([0, 100])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 下图: 储氢SOC ---
    ax = axes[1]
    for d in range(4):
        soc = detailed_results['h2_storage'][:, d] / h2_storage_int * 100 if h2_storage_int > 0 else np.zeros(25)
        ax.plot(hours, soc, '-s', markersize=4, linewidth=2,
                color=season_colors[d], label=season_names[d])
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='SOC下限 5%')
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='SOC上限 95%')
    ax.set_xlabel('时间 (h)')
    ax.set_ylabel('储氢SOC (%)')
    ax.set_title(f'{title_prefix}四季典型日储氢罐SOC变化', fontsize=14, fontweight='bold')
    ax.set_xticks(hours)
    ax.set_xlim([1, 25])
    ax.set_ylim([0, 100])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        filepath = os.path.join(out_dir, 'soc_4seasons.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'SOC变化图已保存: {filepath}')
    return fig


def plot_method_comparison_table(comparison_results, save=True):
    """
    对比实验结果表格图
    输入: comparison_results = [
        {'method': str, 'total_cost': float, 'lpsp': float, 'lreg': float,
         'time': float, 'n_milp_calls': int, 'investment_vars': array},
        ...
    ]
    """
    out_dir = ensure_output_dir()
    fig, ax = plt.subplots(figsize=(16, 4 + len(comparison_results) * 0.6))
    ax.axis('off')

    headers = ['方法', '年化总成本\n(万元/年)', 'LPSP\n(%)', 'LREG\n(%)',
               '求解时间\n(秒)', 'MILP\n调用次数',
               '储能\n(kWh)', '光伏\n(kW)', '风电\n(kW)',
               '电解槽\n(kW)', '储氢\n(kg)', 'FC\n(kW)']

    cell_data = []
    for r in comparison_results:
        cap = r.get('investment_vars', [0]*6)
        row = [
            r['method'],
            f'{r["total_cost"]/1e4:.2f}',
            f'{r["lpsp"]*100:.4f}',
            f'{r["lreg"]*100:.4f}',
            f'{r["time"]:.1f}',
            str(r.get('n_milp_calls', '-')),
            f'{cap[0]:.0f}', f'{cap[1]:.0f}', f'{cap[2]:.0f}',
            f'{cap[3]:.0f}', f'{cap[4]:.0f}', f'{cap[5]:.0f}',
        ]
        cell_data.append(row)

    table = ax.table(cellText=cell_data, colLabels=headers, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # 表头样式
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # 交替行颜色
    for i in range(len(cell_data)):
        color = '#D6E4F0' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(color)

    ax.set_title('对比实验结果汇总', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    if save:
        filepath = os.path.join(out_dir, 'comparison_table.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'对比表格已保存: {filepath}')
    return fig


def plot_cost_breakdown(results_dict, save=True):
    """
    成本构成对比图（投资 vs 运行）
    """
    out_dir = ensure_output_dir()
    methods = list(results_dict.keys())

    inv_costs = [results_dict[m]['investment_cost'] / 1e4 for m in methods]
    op_costs = [results_dict[m]['op_cost'] / 1e4 for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width / 2, inv_costs, width, label='投资成本', color='#4682B4')
    bars2 = ax.bar(x + width / 2, op_costs, width, label='运行成本', color='#CD5C5C')

    for bar, val in zip(bars1, inv_costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, op_costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('方法')
    ax.set_ylabel('成本 (万元/年)')
    ax.set_title('成本构成对比')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        filepath = os.path.join(out_dir, 'cost_breakdown.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'成本构成图已保存: {filepath}')
    return fig
