"""
正负对称的功率平衡堆叠柱状图
更新：包含制氢系统和燃料电池
对应 MATLAB: plot_power_balance_symmetrical.m
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_power_balance_symmetrical(detailed_results, day_to_plot):
    """
    绘制功率平衡对称堆叠图
    输入:
        detailed_results - 详细调度结果字典
        day_to_plot      - 典型日编号 (1-4)
    """
    d = day_to_plot - 1

    p_pv_gen = detailed_results['p_pv'][:, d]
    p_wt_gen = detailed_results['p_wt'][:, d]
    p_dis = detailed_results['p_dis'][:, d]
    p_fc = detailed_results['p_fc'][:, d]
    p_shed = detailed_results['p_shed'][:, d]
    p_cur_pv = detailed_results['p_cur_pv'][:, d]
    p_cur_wt = detailed_results['p_cur_wt'][:, d]

    p_load = detailed_results['p_load'][:, d]
    p_elec = detailed_results['p_elec'][:, d]
    p_ch = detailed_results['p_ch'][:, d]

    total_gen = p_pv_gen + p_wt_gen + p_dis + p_fc + p_shed
    total_con = p_load + p_ch + p_elec
    power_balance = total_gen - total_con

    time_axis = np.arange(1, 25)

    # 颜色
    c_pv   = np.array([44, 160, 44]) / 255
    c_wt   = np.array([31, 119, 180]) / 255
    c_dis  = np.array([255, 127, 14]) / 255
    c_fc   = np.array([255, 165, 0]) / 255
    c_shed = np.array([214, 39, 40]) / 255
    c_cpv  = np.array([255, 215, 0]) / 255
    c_cwt  = np.array([173, 216, 230]) / 255
    c_load = np.array([140, 86, 75]) / 255
    c_elec = np.array([0, 128, 128]) / 255
    c_ch   = np.array([247, 182, 210]) / 255

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # ========== 上半部分：功率平衡图 ==========
    gen_data = np.column_stack([p_pv_gen, p_wt_gen, p_dis, p_fc, p_shed, p_cur_pv, p_cur_wt])
    gen_colors = [c_pv, c_wt, c_dis, c_fc, c_shed, c_cpv, c_cwt]
    gen_labels = ['光伏出力', '风电出力', '储能放电', '燃料电池', '切负荷', '弃光功率', '弃风功率']

    dem_data = np.column_stack([-p_load, -p_elec, -p_ch])
    dem_colors = [c_load, c_elec, c_ch]
    dem_labels = ['用户负荷', '电解槽负荷', '储能充电']

    bottom_gen = np.zeros(24)
    for i in range(gen_data.shape[1]):
        ax1.bar(time_axis, gen_data[:, i], bottom=bottom_gen, width=0.8,
                color=gen_colors[i], label=gen_labels[i])
        bottom_gen += gen_data[:, i]

    bottom_dem = np.zeros(24)
    for i in range(dem_data.shape[1]):
        ax1.bar(time_axis, dem_data[:, i], bottom=bottom_dem, width=0.8,
                color=dem_colors[i], label=dem_labels[i])
        bottom_dem += dem_data[:, i]

    ax1.axhline(y=0, color='k', linewidth=1.5)

    for t in range(24):
        y_top = max(np.sum(gen_data[t, :]), -np.sum(dem_data[t, :])) + 500
        if p_shed[t] > 1e-3:
            ax1.text(t + 1, y_top, '缺电', ha='center', fontweight='bold', color='r')
        elif (p_cur_pv[t] + p_cur_wt[t]) > 1e-3:
            ax1.text(t + 1, y_top, '弃电', ha='center', fontweight='bold', color='b')

    ax1.grid(True)
    ax1.set_title('功率平衡调度图', fontsize=14)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('功率 (kW)')
    ax1.set_xlim([0.5, 24.5])
    ax1.set_xticks(range(2, 25, 2))
    ax1.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.2))

    # ========== 下半部分：功率平衡分析图 ==========
    ax2.plot(time_axis, power_balance, 'k-o', linewidth=2, markerfacecolor='k', label='功率平衡')

    bar_w = 0.6
    ax2.bar(time_axis, p_pv_gen + p_wt_gen, bar_w, color=[0.3, 0.7, 0.3], label='可再生能源', alpha=0.7)

    storage_contrib = p_dis - p_ch
    storage_pos = np.clip(storage_contrib, 0, None)
    storage_neg = np.clip(storage_contrib, None, 0)
    ax2.bar(time_axis, storage_pos, bar_w, color=[1, 0.7, 0], label='储能放电', alpha=0.7)
    ax2.bar(time_axis, storage_neg, bar_w, color=[1, 0.8, 0.8], label='储能充电', alpha=0.7)

    if np.sum(p_fc) > 0:
        ax2.bar(time_axis, p_fc, bar_w, color=[1, 0.5, 0], label='燃料电池', alpha=0.7)

    ax2.bar(time_axis, -p_elec, bar_w, color=[0, 0.5, 0.5], label='电解槽', alpha=0.7)
    ax2.plot(time_axis, -p_load, 'r--', linewidth=2, label='用户负荷')

    ax2.axhline(y=0, color='k', linewidth=1.5)

    # 标记弃电缺电区域
    pb_min = power_balance.min() if power_balance.min() < 0 else -100
    pb_max = power_balance.max() if power_balance.max() > 0 else 100

    for t in range(24):
        if p_shed[t] > 1e-3:
            ax2.axvspan(t + 0.5, t + 1.5, alpha=0.15, color='red')
            ax2.text(t + 1, pb_min * 0.9, '缺电', ha='center', color='r', fontweight='bold')
        if (p_cur_pv[t] + p_cur_wt[t]) > 1e-3:
            ax2.axvspan(t + 0.5, t + 1.5, alpha=0.15, color='blue')
            ax2.text(t + 1, pb_max * 0.9, '弃电', ha='center', color='b', fontweight='bold')

    ax2.grid(True)
    ax2.set_title('功率平衡分析', fontsize=14)
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('功率 (kW)')
    ax2.set_xlim([0.5, 24.5])
    ax2.set_xticks(range(2, 25, 2))
    ax2.legend(loc='best', ncol=3)

    plt.suptitle(f'典型日 {day_to_plot} 功率平衡分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
