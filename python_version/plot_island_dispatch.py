"""
孤岛模式功率调度结果可视化
更新：包含制氢系统和燃料电池，去除充放电功率曲线
对应 MATLAB: plot_island_dispatch.m
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_island_dispatch(detailed_results, day_to_plot):
    """
    绘制孤岛模式功率调度图和能量存储状态图
    输入:
        detailed_results - 详细调度结果字典
        day_to_plot      - 典型日编号 (1-4)
    """
    d = day_to_plot - 1

    # 提取数据
    p_pv_gen = detailed_results['p_pv'][:, d]
    p_wt_gen = detailed_results['p_wt'][:, d]
    p_dis = detailed_results['p_dis'][:, d]
    p_shed = detailed_results['p_shed'][:, d]
    p_cur_pv = detailed_results['p_cur_pv'][:, d]
    p_cur_wt = detailed_results['p_cur_wt'][:, d]
    p_fc = detailed_results['p_fc'][:, d]
    p_load = detailed_results['p_load'][:, d]
    p_ch = detailed_results['p_ch'][:, d]
    p_elec = detailed_results['p_elec'][:, d]

    ee_bat_int = detailed_results['ee_bat_int']
    E_bat_day = detailed_results['E_bat'][:, d]
    soc_percent = E_bat_day / ee_bat_int * 100

    h2_storage_int = detailed_results['h2_storage_int']
    h2_storage_day = detailed_results['h2_storage'][:, d]
    h2_soc_percent = h2_storage_day / h2_storage_int * 100

    time_power = np.arange(1, 25)
    time_soc = np.arange(0, 25)

    # 颜色定义
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

    # ========== 图1: 功率调度图 ==========
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    gen_data = np.column_stack([p_pv_gen, p_wt_gen, p_dis, p_fc, p_shed, p_cur_pv, p_cur_wt])
    gen_colors = [c_pv, c_wt, c_dis, c_fc, c_shed, c_cpv, c_cwt]
    gen_labels = ['光伏出力', '风电出力', '储能放电', '燃料电池', '切负荷', '弃光功率', '弃风功率']

    dem_data = np.column_stack([-p_load, -p_elec, -p_ch])
    dem_colors = [c_load, c_elec, c_ch]
    dem_labels = ['用户负荷', '电解槽负荷', '储能充电']

    bottom_gen = np.zeros(24)
    for i in range(gen_data.shape[1]):
        ax1.bar(time_power, gen_data[:, i], bottom=bottom_gen, width=0.8,
                color=gen_colors[i], label=gen_labels[i])
        bottom_gen += gen_data[:, i]

    bottom_dem = np.zeros(24)
    for i in range(dem_data.shape[1]):
        ax1.bar(time_power, dem_data[:, i], bottom=bottom_dem, width=0.8,
                color=dem_colors[i], label=dem_labels[i])
        bottom_dem += dem_data[:, i]

    ax1.axhline(y=0, color='k', linewidth=1.5)

    # 标记关键时段
    for t in range(24):
        y_top = max(np.sum(gen_data[t, :]), -np.sum(dem_data[t, :])) + 500
        if p_shed[t] > 1e-3:
            ax1.text(t + 1, y_top, '缺电', ha='center', fontweight='bold', color='r')
        elif (p_cur_pv[t] + p_cur_wt[t]) > 1e-3:
            ax1.text(t + 1, y_top, '弃电', ha='center', fontweight='bold', color='b')

    ax1.grid(True)
    ax1.set_title('功率调度图 (孤岛模式)', fontsize=14)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('功率 (kW)')
    ax1.set_xlim([0.5, 24.5])
    ax1.set_xticks(range(0, 25, 2))
    ax1.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.15))

    plt.suptitle(f'典型日 {day_to_plot} 孤岛模式功率调度结果', fontsize=16, fontweight='bold')
    fig1.set_facecolor('white')
    plt.tight_layout()

    # ========== 图2: 能量存储状态图 ==========
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2_left = ax2
    ax2_right = ax2.twinx()

    ax2_left.plot(time_soc, soc_percent, 'm-o', linewidth=2, markersize=5,
                  markerfacecolor='m', label='储能SOC')
    ax2_left.set_ylabel('储能SOC (%)')
    ax2_left.set_ylim([0, 100])

    ax2_right.plot(time_soc, h2_soc_percent, 'c-s', linewidth=2, markersize=5,
                   markerfacecolor='c', label='储氢SOC')
    ax2_right.set_ylabel('储氢SOC (%)')
    ax2_right.set_ylim([0, 100])

    ax2.grid(True)
    ax2.set_title(f'典型日 {day_to_plot} 能量存储状态', fontsize=14)
    ax2.set_xlabel('时间 (小时)')
    ax2.set_xlim([0.5, 24.5])
    ax2.set_xticks(range(0, 25, 2))

    lines1, labels1 = ax2_left.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2_left.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 标记关键时段
    for t in range(24):
        if p_fc[t] > 1e-3:
            ax2_left.text(t + 1, 95, '燃料电池', ha='center', fontweight='bold',
                          color=[1, 0.5, 0], fontsize=8)
        if p_elec[t] > 1e-3:
            ax2_left.text(t + 1, 90, '电解槽', ha='center', fontweight='bold',
                          color=[0, 0.5, 0.5], fontsize=8)

    plt.tight_layout()
