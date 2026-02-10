"""
制氢系统分析可视化函数
功能: 绘制制氢系统的详细运行情况分析图
对应 MATLAB: plot_hydrogen_system_analysis.m
"""
import numpy as np
import matplotlib.pyplot as plt

import optimize_parameters as params


def plot_hydrogen_system_analysis(detailed_results, investment_solution):
    """
    绘制制氢系统分析图
    输入:
        detailed_results    - 详细调度结果字典
        investment_solution - 投资方案 [储能, 光伏, 风电, 电解槽, 储氢, 燃料电池]
    """
    if 'p_elec' not in detailed_results or 'h2_storage' not in detailed_results:
        print('警告: 详细结果中缺少制氢系统数据，无法绘制制氢系统分析图。')
        return

    ee_bat_int = investment_solution[0]
    p_pv_int = investment_solution[1]
    p_wt_int = investment_solution[2]
    p_elec_int = investment_solution[3]
    h2_storage_int = investment_solution[4]

    day_names = ['春季典型日', '夏季典型日', '秋季典型日', '冬季典型日']
    day_colors = ['g', 'r', [1, 0.5, 0], 'b']

    # ========== 图1: 制氢系统功率分析 ==========
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
    fig1.canvas.manager.set_window_title('制氢系统功率分析')

    for day in range(4):
        ax = axes1[day // 2, day % 2]
        hours = np.arange(1, 25)

        p_elec_day = detailed_results['p_elec'][:, day]
        p_fc_day = detailed_results['p_fc'][:, day]
        p_pv_day = detailed_results['p_pv'][:, day]
        p_wt_day = detailed_results['p_wt'][:, day]
        p_load_day = detailed_results['p_load'][:, day]

        ax.plot(hours, p_elec_day, linewidth=2, color=[0, 0.7, 0], label='电解槽功率')
        ax.plot(hours, p_fc_day, linewidth=2, color=[0.8, 0, 0], label='燃料电池功率')
        ax.plot(hours, p_pv_day, '--', linewidth=1.5, color=[1, 0.5, 0], label='光伏出力')
        ax.plot(hours, p_wt_day, '--', linewidth=1.5, color=[0, 0.5, 1], label='风电出力')
        ax.plot(hours, p_load_day, 'k-', linewidth=1, label='负荷需求')
        ax.axhline(y=p_elec_int, color='r', linestyle=':', linewidth=1, label='电解槽额定容量')

        ax.set_xlabel('时间 (h)')
        ax.set_ylabel('功率 (kW)')
        ax.set_title(f'{day_names[day]} - 制氢系统功率')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True)
        ax.set_xlim([1, 24])

    plt.tight_layout()

    # ========== 图2: 储氢系统SOC分析 ==========
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.canvas.manager.set_window_title('储氢系统SOC分析')

    # SOC变化曲线
    for day in range(4):
        hours_ext = np.arange(0, 25)
        h2_soc_day = detailed_results['h2_storage'][:, day] / h2_storage_int * 100
        ax2a.plot(hours_ext, h2_soc_day, linewidth=2, color=day_colors[day], label=day_names[day])

    ax2a.axhline(y=95, color='r', linestyle='--', linewidth=1, label='SOC上限(95%)')
    ax2a.axhline(y=5, color='r', linestyle='--', linewidth=1, label='SOC下限(5%)')
    ax2a.set_xlabel('时间 (h)')
    ax2a.set_ylabel('储氢SOC (%)')
    ax2a.set_title('储氢罐SOC变化')
    ax2a.legend(loc='best')
    ax2a.grid(True)
    ax2a.set_xlim([0, 24])
    ax2a.set_ylim([0, 100])

    # 氢气产量和消耗量
    hours = np.arange(1, 25)
    bar_width = 0.35
    for day in range(4):
        h2_prod_day = detailed_results['h2_prod'][:, day]
        h2_discharge_day = detailed_results['h2_discharge'][:, day]
        x_pos = hours + (day - 1.5) * 0.2

        if day == 0:
            ax2b.bar(x_pos, h2_prod_day, bar_width, color=[0.2, 0.8, 0.2], alpha=0.7, label='氢气产量')
            ax2b.bar(x_pos, -h2_discharge_day, bar_width, color=[0.8, 0.2, 0.2], alpha=0.7, label='氢气消耗')
        else:
            ax2b.bar(x_pos, h2_prod_day, bar_width, color=[0.2, 0.8, 0.2], alpha=0.7)
            ax2b.bar(x_pos, -h2_discharge_day, bar_width, color=[0.8, 0.2, 0.2], alpha=0.7)

    ax2b.set_xlabel('时间 (h)')
    ax2b.set_ylabel('氢气流量 (kg/h)')
    ax2b.set_title('氢气产量与消耗')
    ax2b.legend(loc='best')
    ax2b.grid(True)
    ax2b.set_xlim([0.5, 24.5])

    plt.tight_layout()

    # ========== 图3: 制氢系统经济性分析 ==========
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    fig3.canvas.manager.set_window_title('制氢系统经济性分析')

    N_days_arr = np.array(params.N_days)

    daily_elec_energy = np.sum(detailed_results['p_elec'], axis=0)
    daily_h2_prod = np.sum(detailed_results['h2_prod'], axis=0)
    daily_fc_energy = np.sum(detailed_results['p_fc'], axis=0)

    annual_elec_energy = np.sum(daily_elec_energy * N_days_arr)
    annual_h2_prod = np.sum(daily_h2_prod * N_days_arr)
    annual_fc_energy = np.sum(daily_fc_energy * N_days_arr)

    # 成本构成饼图
    def crf(r, n):
        return r * (r + 1) ** n / ((r + 1) ** n - 1)

    investment_cost_h2 = (crf(params.rp, params.r_elec) * params.c_elec * p_elec_int
                          + crf(params.rp, params.r_h2storage) * params.c_h2storage * h2_storage_int)
    renewable_gen_cost = (p_pv_int * params.cPV + p_wt_int * params.cWT) * 0.05
    om_cost_h2 = annual_elec_energy * params.c_elec_om
    deg_cost = annual_elec_energy * params.c_elec_deg

    cost_values = [investment_cost_h2, renewable_gen_cost, om_cost_h2, deg_cost]
    cost_labels = ['投资成本', '电力成本', '运维成本', '退化成本']
    axes3[0, 0].pie(cost_values, labels=cost_labels, autopct='%1.1f%%')
    axes3[0, 0].set_title('制氢系统年化成本构成')

    # 能量流动
    energy_flow = [annual_elec_energy, annual_h2_prod * 33.33, annual_fc_energy]
    energy_labels = ['电解槽耗电', '氢能储存', '燃料电池发电']
    axes3[0, 1].bar(range(3), energy_flow, color=[0.3, 0.6, 0.9])
    axes3[0, 1].set_xticks(range(3))
    axes3[0, 1].set_xticklabels(energy_labels)
    axes3[0, 1].set_ylabel('年能量 (kWh)')
    axes3[0, 1].set_title('制氢系统年能量流动')
    axes3[0, 1].grid(True)

    # 制氢效率
    daily_efficiency = np.zeros(4)
    for day in range(4):
        elec_sum = np.sum(detailed_results['p_elec'][:, day])
        if elec_sum > 0:
            daily_efficiency[day] = np.sum(detailed_results['h2_prod'][:, day]) * 33.33 / elec_sum

    axes3[1, 0].bar(range(4), daily_efficiency * 100, color=[0.9, 0.6, 0.3])
    axes3[1, 0].set_xticks(range(4))
    axes3[1, 0].set_xticklabels(['春', '夏', '秋', '冬'])
    axes3[1, 0].set_ylabel('制氢效率 (%)')
    axes3[1, 0].set_title('各季节制氢效率')
    axes3[1, 0].set_ylim([0, 80])
    axes3[1, 0].grid(True)

    # 设备利用率
    elec_utilization = daily_elec_energy / (p_elec_int * 24) * 100
    axes3[1, 1].bar(range(4), elec_utilization, color=[0.6, 0.9, 0.3])
    axes3[1, 1].set_xticks(range(4))
    axes3[1, 1].set_xticklabels(['春', '夏', '秋', '冬'])
    axes3[1, 1].set_ylabel('设备利用率 (%)')
    axes3[1, 1].set_title('电解槽设备利用率')
    axes3[1, 1].set_ylim([0, 100])
    axes3[1, 1].grid(True)

    plt.tight_layout()

    # ========== 图4: 制氢-可再生能源耦合分析 ==========
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.canvas.manager.set_window_title('制氢-可再生能源耦合分析')

    renewable_gen = detailed_results['p_pv'] + detailed_results['p_wt']
    renewable_curtail = detailed_results['p_cur_pv'] + detailed_results['p_cur_wt']
    renewable_to_h2 = detailed_results['p_elec']

    daily_re_total = np.sum(renewable_gen, axis=0)
    daily_re_curtail = np.sum(renewable_curtail, axis=0)
    daily_re_to_h2 = np.sum(renewable_to_h2, axis=0)
    daily_re_utilized = daily_re_total - daily_re_curtail

    x = np.arange(4)
    ax4a.bar(x, daily_re_total, 0.6, color=[0.8, 0.8, 0.8], label='可再生能源总量')
    ax4a.bar(x, daily_re_utilized, 0.6, color=[0.2, 0.8, 0.2], label='有效利用')
    ax4a.bar(x, daily_re_to_h2, 0.6, color=[0.2, 0.2, 0.8], label='制氢消纳')
    ax4a.set_xticks(x)
    ax4a.set_xticklabels(['春', '夏', '秋', '冬'])
    ax4a.set_ylabel('日发电量 (kWh)')
    ax4a.set_title('可再生能源消纳情况')
    ax4a.legend(loc='best')
    ax4a.grid(True)

    # 灵活性指标
    flexibility = np.zeros((3, 4))
    for day in range(4):
        baseline_curtail = max(0, daily_re_total[day] - np.sum(detailed_results['p_load'][:, day]))
        actual_curtail = daily_re_curtail[day]
        if baseline_curtail > 0:
            flexibility[0, day] = (baseline_curtail - actual_curtail) / baseline_curtail * 100

        elec_running = detailed_results['p_elec'][:, day]
        elec_running_pos = elec_running[elec_running > 0]
        if len(elec_running_pos) > 0:
            flexibility[1, day] = (elec_running_pos.max() - elec_running_pos.min()) / p_elec_int * 100

        max_fc = np.max(detailed_results['p_fc'][:, day])
        max_load = np.max(detailed_results['p_load'][:, day])
        if max_load > 0:
            flexibility[2, day] = max_fc / max_load * 100

    x = np.arange(4)
    w = 0.25
    ax4b.bar(x - w, flexibility[0], w, label='弃电减少率')
    ax4b.bar(x, flexibility[1], w, label='负荷调节能力')
    ax4b.bar(x + w, flexibility[2], w, label='应急供电能力')
    ax4b.set_xticks(x)
    ax4b.set_xticklabels(['春', '夏', '秋', '冬'])
    ax4b.set_ylabel('灵活性指标 (%)')
    ax4b.set_title('制氢系统灵活性贡献')
    ax4b.legend(loc='best')
    ax4b.grid(True)

    plt.tight_layout()

    # 输出关键指标
    print('\n=== 制氢系统关键指标 ===')
    print(f'年制氢量: {annual_h2_prod:.1f} kg')
    print(f'年平均制氢效率: {np.mean(daily_efficiency) * 100:.2f}%')
    print(f'电解槽年平均利用率: {np.mean(elec_utilization):.2f}%')
    print(f'制氢系统年化投资成本: {investment_cost_h2:.0f} 元')
    if annual_h2_prod > 0:
        print(f'制氢单位成本: {sum(cost_values) / annual_h2_prod:.2f} 元/kg')
