"""
并网模式微电网优化参数设置 - 包含制氢系统
对应 MATLAB: optimize_paremeters.m
"""

# ==================== 投资成本参数 ====================
rp = 0.08  # 贴现率

# 投资年限
rbat = 12          # 储能
rPV = 25           # 光伏
rWT = 20           # 风电
r_elec = 15        # 电解槽
r_h2storage = 20   # 储氢设备
r_fc = 21          # 燃料电池

Imax = 50  # 迭代次数

# 单位容量投资成本
cbat = 1500         # 储能投资成本 (元/kWh)
cPV = 4000          # 光伏投资成本 (元/kW)
cWT = 7000          # 风电投资成本 (元/kW)
c_elec = 5000       # 电解槽投资成本 (元/kW)
c_h2storage = 10000 # 储氢罐投资成本 (元/kg)
c_fc = 4500         # 燃料电池投资成本 (元/kW)

eta = 0.95  # 储能充放电效率

# ==================== 运维成本系数 ====================
c_wt_om = 0.0296    # 风电运维成本 (元/kWh)
c_pv_om = 0.0096    # 光伏运维成本 (元/kWh)
c_bat_om = 0.05     # 储能运维成本 (元/kWh)
c_bat_deg = 0.2     # 储能退化成本 (元/kWh)

# 制氢系统运维成本
c_elec_om = 0.015       # 电解槽运维成本 (元/kWh)
c_elec_deg = 0.05       # 电解槽退化成本 (元/kWh)
c_fc_om = 0.020         # 燃料电池运维成本 (元/kWh)
c_fc_deg = 0.030        # 燃料电池退化成本 (元/kWh)
c_fc_startup = 50       # 燃料电池启动成本 (元)
c_h2storage_om = 0.001  # 储氢罐运维成本 (元/kg/day)
c_h2_leakage = 30       # 氢气泄漏成本 (元/kg)
c_h2_storage = 0.001    # 氢气储存持有成本 (元/kg/h)
c_shed = 30             # 缺电惩罚成本 (元/kWh)

# 方案2：下层调度惩罚系数（用于规范调度逻辑，不计入上层成本目标）
c_shed_penalty = 10     # 缺电惩罚系数 (元/kWh)
c_curt_penalty = 0.1    # 弃电惩罚系数 (元/kWh)

# 电解槽启动成本
elec_startup_cost = 25  # 电解槽启动成本 (元/次)

M = 100000  # 一个极大正数系数

# 可靠性硬约束参数
MAX_LPSP_ALLOWED = 0.05  # 最大允许缺电率 (5%)

# 储能电池参数
bat_cycle_life = 3000  # 电池循环寿命次数

# ==================== 制氢系统技术参数 ====================
# 电解槽参数
elec_efficiency_rated = 0.65  # 额定效率
elec_min_load = 0.20          # 最小负荷率
elec_max_load = 1.00          # 最大负荷率
elec_ramp_up = 0.5            # 爬坡速率 (额定功率/h)
elec_ramp_down = 0.8          # 降坡速率 (额定功率/h)
LHV_H2 = 33.33               # 氢气低位发热值 (kWh/kg)

# 储氢系统参数
h2_storage_efficiency = 0.95  # 储存效率
h2_leakage_rate = 0.0002      # 日泄漏率 (0.02%/day)
h2_soc_min = 0.05             # 最小SOC
h2_soc_max = 0.95             # 最大SOC

# 燃料电池参数
fc_efficiency = 0.50          # 发电效率
fc_min_load = 0.30            # 最小负荷率
fc_h2_consumption = 0.06      # 氢气消耗率 (kg/kWh)
fc_ramp_up = 0.8              # 爬坡速率 (额定功率/h)
fc_ramp_down = 1.0            # 降坡速率 (额定功率/h)

# 典型日天数
N_days = [91, 92, 91, 91]  # 春夏秋冬四个典型日的天数

# ==================== 离网模式参数 ====================
# 归一化参考基准（离网模式）
C_ref_offgrid = 2.5e7   # 离网模式参考成本 (元/年)
LPSP_ref = 0.05          # LPSP归一化参考值 (5%)
LREG_ref = 0.15          # LREG归一化参考值 (15%)
