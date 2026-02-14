"""
双层优化统一参数配置
上层: DRL-PPO容量规划
下层: CPLEX MILP运行优化
融合 optimize_parameters.py + PPO超参数
"""
import os

# ==================== 数据路径 ====================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
EXCEL_FILE = os.path.join(DATA_DIR, '四个典型日数据.xlsx')

# ==================== 投资成本参数 ====================
rp = 0.08          # 贴现率

# 投资年限
rbat = 12           # 储能
rPV = 25            # 光伏
rWT = 20            # 风电
r_elec = 15         # 电解槽
r_h2storage = 20    # 储氢设备
r_fc = 21           # 燃料电池

# 单位容量投资成本 (元/kWh 或 元/kW 或 元/kg)
cbat = 1500         # 储能投资成本 (元/kWh)
cPV = 4000          # 光伏投资成本 (元/kW)
cWT = 7000          # 风电投资成本 (元/kW)
c_elec = 5000       # 电解槽投资成本 (元/kW)
c_h2storage = 10000 # 储氢罐投资成本 (元/kg)
c_fc = 4500         # 燃料电池投资成本 (元/kW)

eta = 0.95          # 储能充放电效率

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

# VoLL经济化处理（替代原惩罚系数，消除上下层目标不一致）
VoLL = 30.0             # 切负荷价值 (元/kWh) — Value of Lost Load
c_curtail_opp = 0.5     # 弃电机会成本 (元/kWh)

# 电解槽启动成本
elec_startup_cost = 25  # 电解槽启动成本 (元/次)

M = 100000  # 大M系数

# ==================== 可靠性约束参数 ====================
MAX_LPSP_ALLOWED = 0.05  # 最大允许缺电率 (5%)
MAX_LREG_ALLOWED = 0.15  # 最大允许弃电率 (15%)

# 储能电池参数
bat_cycle_life = 3000

# ==================== 制氢系统技术参数 ====================
# 电解槽参数
elec_efficiency_rated = 0.65
elec_min_load = 0.20
elec_max_load = 1.00
elec_ramp_up = 0.5      # 爬坡速率 (额定功率/h)
elec_ramp_down = 0.8     # 降坡速率 (额定功率/h)
LHV_H2 = 33.33           # 氢气低位发热值 (kWh/kg)

# 储氢系统参数
h2_storage_efficiency = 0.95
h2_leakage_rate = 0.0002  # 日泄漏率 (0.02%/day)
h2_soc_min = 0.05
h2_soc_max = 0.95

# 燃料电池参数
fc_efficiency = 0.50
fc_min_load = 0.30
fc_h2_consumption = 0.06  # kg/kWh
fc_ramp_up = 0.8
fc_ramp_down = 1.0

# 典型日天数
N_days = [91, 92, 91, 91]  # 春夏秋冬

# ==================== 归一化参考基准 ====================
C_ref = 2.5e7       # 参考成本 (元/年)
LPSP_ref = 0.05      # LPSP归一化参考值
LREG_ref = 0.15      # LREG归一化参考值

# ==================== 容量规划决策空间 ====================
# 6维决策变量: [储能(kWh), 光伏(kW), 风电(kW), 电解槽(kW), 储氢(kg), 燃料电池(kW)]
CAP_LOWER = [7000, 8000, 5000, 3500, 1000, 2000]
CAP_UPPER = [16000, 12000, 12000, 8000, 6000, 8000]
N_CAP_VARS = 6
CAP_NAMES = ['储能(kWh)', '光伏(kW)', '风电(kW)', '电解槽(kW)', '储氢(kg)', '燃料电池(kW)']

# ==================== PPO超参数 ====================
RANDOM_SEED = 42
TRAIN_EPISODES = 600      # 训练回合数（调试阶段，正式训练恢复1000-1500）
TEST_EPISODES = 1          # 测试回合数
N_STEPS_PER_EPISODE = 50   # 每episode最大步数（容量搜索步）
GAMMA = 0.99               # 折扣因子
LR_ACTOR = 1e-4            # Actor学习率
LR_CRITIC = 2e-4           # Critic学习率
ACTOR_UPDATE_STEPS = 10    # Actor更新步数
CRITIC_UPDATE_STEPS = 10   # Critic更新步数
EPSILON_CLIP = 0.2         # PPO-Clip范围
GAE_LAMBDA = 0.95          # GAE λ
ENTROPY_COEF = 0.01        # 熵正则系数
HIDDEN_DIM = 256           # 隐藏层维度
GRAD_CLIP = 0.5            # 梯度裁剪

# 容量搜索步长（每步最大调整量 = (上限-下限) * STEP_RATIO）
STEP_RATIO = 0.10
NO_IMPROVE_PATIENCE = 15  # Episode内连续无改进步数达到此值则提前终止
CACHE_GRID_RATIO = 0.01   # 容量评估缓存网格（容量范围的1%）

# ==================== Reward设计参数 ====================
REWARD_SCALE = 1e7         # 奖励归一化尺度
LAMBDA_LPSP = 5000.0       # LPSP约束惩罚系数
LAMBDA_LREG = 500.0        # LREG约束惩罚系数
INFEASIBLE_PENALTY = -100.0  # 不可行解大惩罚

# ==================== 求解器配置 ====================
SOLVER_NAME = 'cplex'
SOLVER_TIMELIMIT = 15      # 秒（训练阶段不需高精度）
SOLVER_MIPGAP = 0.05       # 训练阶段5%间隙足够

# ==================== 输出路径 ====================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
