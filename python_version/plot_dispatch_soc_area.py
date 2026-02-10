"""
24小时功率调度与储能状态可视化
功能：分别绘制功率堆叠图和储能状态图，学术风格
对应 MATLAB: plot_dispatch_soc_area.m
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_dispatch_soc_area(detailed_results, day_to_plot):
    """
    绘制指定典型日的功率调度堆叠图和储能SOC图
    输入:
        detailed_results - 详细调度结果字典
        day_to_plot      - 典型日编号 (1-4: 春夏秋冬)
    """
    d = day_to_plot - 1  # 转为0-indexed

    # ========== 1. 提取数据 ==========
    p_pv_gen = detailed_results['p_pv'][:, d]
    p_wt_gen = detailed_results['p_wt'][:, d]
    p_dis = detailed_results['p_dis'][:, d]
    p_fc = detailed_results['p_fc'][:, d]
    p_shed = detailed_results['p_shed'][:, d]

    p_load = detailed_results['p_load'][:, d]
    p_ch = detailed_results['p_ch'][:, d]
    p_elec = detailed_results['p_elec'][:, d]

    p_cur_pv = detailed_results['p_cur_pv'][:, d]
    p_cur_wt = detailed_results['p_cur_wt'][:, d]

    ee_bat_int = detailed_results['ee_bat_int']
    E_bat_day = detailed_results['E_bat'][:, d]  # 25 points (0-24h)

    h2_storage_int = detailed_results['h2_storage_int']
    h2_storage_day = detailed_results['h2_storage'][:, d]  # 25 points

    soc_percent = E_bat_day / ee_bat_int * 100
    h2_soc_percent = h2_storage_day / h2_storage_int * 100

    time_power = np.arange(1, 25)
    time_soc = np.arange(0, 25)

    # ========== 2. 功率堆叠图 ==========
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    fig1.canvas.manager.set_window_title(f'典型日{day_to_plot} 功率调度堆叠图')

    # 颜色方案
    colors_gen = {
        'pv': [0.2, 0.6, 0.2],
        'wt': [0.2, 0.4, 0.8],
        'dis': [0.8, 0.6, 0.0],
        'fc': [0.8, 0.4, 0.1],
        'shed': [0.6, 0.0, 0.0],
        'cur_pv': [0.5, 0.5, 0.5],
        'cur_wt': [0.6, 0.6, 0.6],
    }
    colors_con = {
        'load': [0.4, 0.2, 0.2],
        'ch': [0.4, 0.4, 0.6],
        'elec': [0.1, 0.5, 0.4],
    }

    # 发电侧堆叠
    gen_data = np.column_stack([p_pv_gen, p_wt_gen, p_dis, p_fc, p_shed, p_cur_pv, p_cur_wt])
    gen_labels = ['光伏发电', '风力发电', '储能放电', '燃料电池', '切负荷', '弃光功率', '弃风功率']
    gen_colors = [colors_gen['pv'], colors_gen['wt'], colors_gen['dis'], colors_gen['fc'],
                  colors_gen['shed'], colors_gen['cur_pv'], colors_gen['cur_wt']]

    # 用电侧堆叠（负值）
    con_data = np.column_stack([-p_load, -p_ch, -p_elec])
    con_labels = ['用电负荷', '储能充电', '电解槽']
    con_colors = [colors_con['load'], colors_con['ch'], colors_con['elec']]

    # 绘制发电侧
    bottom_gen = np.zeros(24)
    bars_gen = []
    for i in range(gen_data.shape[1]):
        b = ax1.bar(time_power, gen_data[:, i], bottom=bottom_gen, width=0.9,
                    color=gen_colors[i], edgecolor='none', alpha=0.95, label=gen_labels[i])
        bars_gen.append(b)
        bottom_gen += gen_data[:, i]

    # 绘制用电侧
    bottom_con = np.zeros(24)
    bars_con = []
    for i in range(con_data.shape[1]):
        b = ax1.bar(time_power, con_data[:, i], bottom=bottom_con, width=0.9,
                    color=con_colors[i], edgecolor='none', alpha=0.95, label=con_labels[i])
        bars_con.append(b)
        bottom_con += con_data[:, i]

    ax1.axhline(y=0, color='k', linewidth=2.5, alpha=0.8)
    ax1.set_xlim([0.5, 24.5])
    ax1.set_xticks(range(2, 25, 2))
    ax1.set_xlabel('时间 (h)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('功率 (kW)', fontsize=14, fontweight='bold')
    ax1.set_title('24小时功率调度堆叠图', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.15),
               edgecolor=[0.2, 0.2, 0.2])

    # 统计信息
    total_gen = np.sum(p_pv_gen + p_wt_gen + p_dis + p_fc + p_shed)
    total_con = np.sum(p_load + p_ch + p_elec)
    total_curt = np.sum(p_cur_pv + p_cur_wt)
    total_shed_energy = np.sum(p_shed)
    lreg_day = total_curt / (total_gen + total_curt) * 100 if (total_gen + total_curt) > 0 else 0
    lpsp_day = total_shed_energy / np.sum(p_load) * 100 if np.sum(p_load) > 0 else 0

    info_text = (f'能量统计\n发电量: {total_gen:.1f} kWh\n用电量: {total_con:.1f} kWh\n'
                 f'弃电量: {total_curt:.1f} kWh\n缺电量: {total_shed_energy:.1f} kWh\n'
                 f'LREG: {lreg_day:.2f}%\nLPSP: {lpsp_day:.2f}%')
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor=[0.3, 0.3, 0.3]))

    fig1.set_facecolor('white')
    plt.tight_layout()

    # ========== 3. 储能状态图 ==========
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.canvas.manager.set_window_title(f'典型日{day_to_plot} 储能状态')

    ax2_left = ax2
    ax2_right = ax2.twinx()

    line1, = ax2_left.plot(time_soc, soc_percent, 'o-', color=[0.8, 0.2, 0.6],
                           linewidth=2.5, markersize=6, markerfacecolor=[0.8, 0.2, 0.6],
                           markeredgecolor='white', label='储能SOC')
    ax2_left.set_ylabel('储能SOC (%)', fontsize=14, fontweight='bold', color=[0.8, 0.2, 0.6])
    ax2_left.set_ylim([0, 100])
    ax2_left.tick_params(axis='y', labelcolor=[0.8, 0.2, 0.6])

    line2, = ax2_right.plot(time_soc, h2_soc_percent, 's-', color=[0.2, 0.6, 0.8],
                            linewidth=2.5, markersize=6, markerfacecolor=[0.2, 0.6, 0.8],
                            markeredgecolor='white', label='储氢SOC')
    ax2_right.set_ylabel('储氢SOC (%)', fontsize=14, fontweight='bold', color=[0.2, 0.6, 0.8])
    ax2_right.set_ylim([0, 100])
    ax2_right.tick_params(axis='y', labelcolor=[0.2, 0.6, 0.8])

    ax2_left.set_xlim([1, 24])
    ax2_left.set_xticks(range(2, 25, 2))
    ax2_left.set_xlabel('时间 (h)', fontsize=14, fontweight='bold')
    ax2_left.set_title('储能系统状态变化', fontsize=16, fontweight='bold')
    ax2_left.grid(True, alpha=0.3)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax2_left.legend(lines, labels, loc='best', fontsize=12)

    fig2.set_facecolor('white')
    plt.tight_layout()

    # ========== 4. 功率平衡验证 ==========
    print(f'\n=== 典型日 {day_to_plot} 功率平衡验证 (离网模式) ===')
    generation = p_pv_gen + p_wt_gen + p_dis + p_fc + p_shed
    demand = p_load + p_ch + p_elec
    power_balance = generation - demand
    max_imbalance = np.max(np.abs(power_balance))

    if max_imbalance < 1e-3:
        print(f'[PASS] 功率平衡验证通过，最大不平衡: {max_imbalance:.6f} kW')
    else:
        print(f'[FAIL] 功率平衡验证失败，最大不平衡: {max_imbalance:.3f} kW')

    print(f'总能量统计: 发电{total_gen:.1f} kWh, 用电{total_con:.1f} kWh, 弃电{total_curt:.1f} kWh')
