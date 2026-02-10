"""
环境要求:
python = 3.8
pytorch= 2.2.1
gym = 0.11.0
pandas= 2.0.3
matplotlib = 3.4.3
numpy =1.24.3
"""

# 导入所需的库
import argparse  # 用于命令行参数解析
import os       # 操作系统接口
import time     # 时间相关函数
import matplotlib.pyplot as plt  # 绘图库
import matplotlib as mpl        # matplotlib配置
import numpy as np             # 数值计算库
import pandas as pd           # 数据处理库
import gym                    # 强化学习环境库
from gym.utils import seeding  # gym随机数生成
from gym import spaces        # gym空间定义
import torch                  # PyTorch深度学习框架
import torch.nn as nn        # 神经网络模块
import torch.nn.functional as F  # 激活函数等功能模块
from torch.distributions import Normal  # 正态分布

######################################   负荷与风光出力  ####################################
# 从Excel文件读取24小时负荷和风光数据
load = pd.read_excel(pd.ExcelFile("1hdata.xlsx"))
p_load_24h0 = list(load.iloc[:, 2])  # 电负荷数据
p_wt_24h0 = list(load.iloc[:, 3])    # 风电出力数据
p_pv_24h0 = list(load.iloc[:, 4])    # 光伏出力数据

# 初始化全局变量（将在Environment.reset中更新）
p_load_24h = list(p_load_24h0)
p_wt_24h = list(p_wt_24h0)
p_pv_24h = list(p_pv_24h0)

#####################################  购电/售电电价与气价  ######################################
# 初始化购电和售电价格列表(25小时,包含第二天0时刻)
price_ele_buy = []   # 购电价格列表
price_ele_sell = []  # 售电价格列表

# 设置分时电价（参考刘沛津文献表C3）
for i in range(25):
    if i < 7:  # 谷时段(0-7点)
        price_ele_buy.append(0.44)
        price_ele_sell.append(0.24)
    if 7 <=  i < 12:  # 平时段(7-12点)
        price_ele_buy.append(0.67)
        price_ele_sell.append(0.51)
    if 12 <= i < 19:  # 峰时段(12-19点)
        price_ele_buy.append(0.88)
        price_ele_sell.append(0.75)
    if 19 <= i < 23:  # 平时段(19-23点)
        price_ele_buy.append(0.67)
        price_ele_sell.append(0.51)
    if 23 <= i <= 24:  # 谷时段(23-24点)
        price_ele_buy.append(0.44)
        price_ele_sell.append(0.24)

# 价格放大1000倍
price_ele_buy = [i * 1000 for i in price_ele_buy]
price_ele_sell = [i * 1000 for i in price_ele_sell]

#####################################  售氢与碳交易参数  ######################################
price_h2 = 30                  # 氢气售价 30元/kg
h2_sell_rate_max = 0.03        # 最大售氢速率（储氢容量的5%/h）
P_CO2 = 0.07                   # 碳交易价格 元/kg_CO2（不乘1000）
c_e = 0.5703                   # 电网碳排放因子 kg_CO2/kWh
c_H2 = 20                      # 氢气碳减排系数 kg_CO2/kg_H2
c_CO2 = 0.941                  # 燃料电池碳减排系数 kg_CO2/kWh

##################### 超参数设置 #########################################
RANDOM_SEED = 42   # 随机种子,用于结果复现
RENDER = False  # 是否渲染环境

# 添加成本权重系数
COST_WEIGHT = 1.0      # 实际成本的权重系数
PUNISHMENT_WEIGHT = 1.0 # 惩罚项的权重系数

ALG_NAME = 'PPO'  # 算法名称
TRAIN_EPISODES = 5000  # 训练回合数
TEST_EPISODES = 1  # 测试回合数
MAX_STEPS = 24  # 每个回合的最大步数(对应24小时)
TEST_MAX_STEPS = 24  # 测试时的最大步数
GAMMA = 0.98  # 奖励折扣因子
LR_A = 0.0001  # Actor网络学习率
LR_C = 0.0002  # Critic网络学习率
BATCH_SIZE = 24  # 批次大小
ACTOR_UPDATE_STEPS = 10  # Actor网络更新步数
CRITIC_UPDATE_STEPS = 10  # Critic网络更新步数

# PPO-penalty参数
KL_TARGET = 0.01  # KL散度目标值
LAM = 0.5  # GAE-Lambda参数

# PPO-clip参数
EPSILON = 0.2  # 裁剪范围(可以适当增大到0.3，增加探索范围)

######################################### 环境类定义 #################################
class Environment(gym.Env):
    def __init__(self):
        """
        初始化函数: 设置各种设备的运行参数
        """
        # 电网交换参数
        self.min_p_grid = -3  # 系统与主电网最小交换功率/MW(负值表示向电网售电)
        self.max_p_grid = 4   # 系统与主电网最大交换功率/MW
        
        # 电储能参数
        self.min_p_bes = -1.5 # 电储能最小出力/MW(负值表示充电)
        self.max_p_bes = 1.5  # 电储能最大出力/MW(正值表示放电)
        self.min_c_soc = 0.1  # 电储能最小荷电状态
        self.max_c_soc = 0.9  # 电储能最大荷电状态
        self.q_bes = 6        # 电储能容量/MWh
        self.n_ch = 0.95      # 电储能充电效率
        self.n_dis = 0.95     # 电储能放电效率
        
        # 电解槽参数
        self.min_p_elec = 0   # 电解槽最小功率/MW
        self.max_p_elec = 2   # 电解槽最大功率/MW
        self.elec_efficiency = 0.65  # 电解槽效率
        self.elec_min_load = 0.2     # 电解槽最小负荷率
        self.LHV_H2 = 33.33   # 氢气低位发热值 kWh/kg
        
        # 储氢罐参数
        self.h2_storage_cap = 500    # 储氢容量/kg
        self.h2_storage_eff = 0.95   # 储存效率
        self.min_h2_soc = 0.1        # 储氢最小SOC
        self.max_h2_soc = 0.9        # 储氢最大SOC
        
        # 燃料电池参数
        self.min_p_fc = 0     # 燃料电池最小功率/MW
        self.max_p_fc = 2     # 燃料电池最大功率/MW
        self.fc_efficiency = 0.60    # 燃料电池效率
        self.fc_min_load = 0.2       # 燃料电池最小负荷率
        
        # 时间参数
        self.time_step = 1    # 时间步长为1小时
        self.time_min = 0     # 最小时间(0时)
        self.time_max = 24    # 最大时间(24时)

        # 定义动作空间(4个连续动作:p_elec电解槽、p_bes储能、p_fc燃料电池、h2_sell售氢)
        self.max_h2_sell = self.h2_storage_cap * h2_sell_rate_max
        self.min_h2_sell = 0
        high = np.array([self.max_p_elec, self.max_p_bes, self.max_p_fc, self.max_h2_sell], dtype=np.float32)
        low = np.array([self.min_p_elec, self.min_p_bes, self.min_p_fc, self.min_h2_sell], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 定义状态空间(14维，归一化)
        # [p_load, p_wt, p_pv, c_soc, h2_soc, sin_t, cos_t, price_buy_norm, price_sell_norm, 
        #  last_p_elec_norm, last_p_fc_norm, elec_on, fc_on, net_load_norm]
        self.obs_dim = 14
        high1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 1, 1.0], dtype=np.float32)
        low1 = np.array([0, 0, 0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0, 0, -1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low1, high=high1, dtype=np.float32)
        
        # 价格归一化参数
        self.price_buy_max = max(price_ele_buy)
        self.price_sell_max = max(price_ele_sell)

        # 启停状态记录
        self.elec_was_on = False
        self.fc_was_on = False

        # 启动成本（元/次）
        self.elec_startup_cost = 25.0
        self.fc_startup_cost = 50.0

        # 爬坡约束参数
        self.elec_ramp_up = 0.5    # 电解槽上爬坡率 (额定功率/h)
        self.elec_ramp_down = 0.8  # 电解槽下爬坡率
        self.fc_ramp_up = 0.8      # 燃料电池上爬坡率
        self.fc_ramp_down = 1.0    # 燃料电池下爬坡率

        # 上一时刻功率记录
        self.last_p_elec = 0
        self.last_p_fc = 0

        # 退化成本系数 (元/kWh)
        self.c_bat_deg = 0.1       # 储能退化 100元/MWh
        self.c_elec_deg = 0.05     # 电解槽退化 50元/MWh
        self.c_fc_deg = 0.03       # 燃料电池退化 30元/MWh

        # 分时段策略惩罚参数
        self.peak_elec_penalty = 100     # 峰时段制氢惩罚 元/MW
        self.valley_elec_bonus = 50      # 谷时段制氢奖励 元/MW

        # 缺电惩罚参数
        self.shortage_penalty = 10.0     # 缺电惩罚系数（相对于punishment_value的倍数）

        # 观测归一化参数（留安全裕度）
        self.P_SCALE = 7.0
        self.NET_SCALE = 4.0

        # 日终SOC闭合惩罚参数
        self.k_terminal_soc = 1000.0
        self.k_terminal_h2 = 8000.0

        # SOC闭合目标（在reset时更新）
        self.soc_init = 0.5
        self.h2_soc_init = 0.5

        # ==================== 可达性漏斗(Reachability Funnel)参数 ====================
        # 训练模式开关：训练时关闭t==23硬闭合，让策略学习闭合
        self.training_mode = True
        
        # 可达性漏斗惩罚系数（时变：越接近终点权重越大）
        # 注意：惩罚会被REWARD_SCALE=5000归一化，需要足够大才能产生有效信号
        self.k_funnel_base = 30000.0     # 基础惩罚系数（归一化后约1-2量级）
        self.funnel_alpha = 2.0         # 时变指数 k(t) = k_base * (1 + (t/23)^alpha)
        
        # SOC闭合容差（允许的终端偏差）
        self.soc_tolerance = 0.02       # 电池SOC容差 2%
        self.h2_soc_tolerance = 0.02    # 储氢SOC容差 2%

        # 设置随机种子
        self.seed()

    def seed(self, seed=None):
        """
        设置随机数种子,用于复现结果
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reachability_penalty(self, new_c_soc, new_h2_soc, t):
        """
        计算可达性漏斗惩罚：检查剩余时间内能否回到目标SOC
        参数:
            new_c_soc: 当前步执行后的电池SOC
            new_h2_soc: 当前步执行后的储氢SOC
            t: 当前时刻（0-23，执行动作时的时刻）
        返回:
            penalty: 不可达惩罚值
            funnel_info: 调试信息字典
        """
        # 剩余可调度时间（执行完当前步后还剩几小时）
        rem = 23 - t
        if rem <= 0:
            # 最后一步，直接用实际偏差作为惩罚（不再返回0）
            soc_err = max(0, abs(new_c_soc - self.soc_init) - self.soc_tolerance)
            h2_err = max(0, abs(new_h2_soc - self.h2_soc_init) - self.h2_soc_tolerance)
            # 使用最大权重（t=23时 time_weight = 2）
            k_funnel = self.k_funnel_base * 2
            penalty = k_funnel * (soc_err ** 2 + h2_err ** 2)
            return penalty, {'soc_err': soc_err, 'h2_err': h2_err, 'soc_unreachable': soc_err, 'h2_unreachable': h2_err, 'time_weight': 2, 'k_funnel': k_funnel}
        
        # ==================== 电池SOC可达性 ====================
        # 剩余时间内最大可充电量（SOC上升）
        delta_soc_max_up = rem * self.n_ch * self.max_p_bes / self.q_bes
        # 剩余时间内最大可放电量（SOC下降）
        delta_soc_max_down = rem * self.max_p_bes / self.n_dis / self.q_bes
        
        # 考虑SOC边界约束后的实际可达范围
        soc_reachable_max = min(self.max_c_soc, new_c_soc + delta_soc_max_up)
        soc_reachable_min = max(self.min_c_soc, new_c_soc - delta_soc_max_down)
        
        # 目标SOC与可达范围的偏差
        soc_target = self.soc_init
        if soc_target > soc_reachable_max:
            soc_unreachable = soc_target - soc_reachable_max  # 需要充得太多，不可达
        elif soc_target < soc_reachable_min:
            soc_unreachable = soc_reachable_min - soc_target  # 需要放得太多，不可达
        else:
            soc_unreachable = 0  # 目标在可达范围内
        
        # ==================== 储氢SOC可达性 ====================
        # 最大产氢速率（电解槽满功率，使用标称效率估算）
        max_h2_prod_rate = self.max_p_elec * self.elec_efficiency / self.LHV_H2 * 1000 / self.h2_storage_cap
        # 最大耗氢速率（燃料电池满功率 + 最大售氢）
        max_h2_consume_rate = (self.max_p_fc / self.fc_efficiency / self.LHV_H2 * 1000 + self.max_h2_sell) / self.h2_storage_cap
        
        # 剩余时间内最大可增/减的H2 SOC
        delta_h2_max_up = rem * max_h2_prod_rate
        delta_h2_max_down = rem * max_h2_consume_rate
        
        # 考虑SOC边界约束后的实际可达范围
        h2_reachable_max = min(self.max_h2_soc, new_h2_soc + delta_h2_max_up)
        h2_reachable_min = max(self.min_h2_soc, new_h2_soc - delta_h2_max_down)
        
        # 目标H2 SOC与可达范围的偏差
        h2_target = self.h2_soc_init
        if h2_target > h2_reachable_max:
            h2_unreachable = h2_target - h2_reachable_max
        elif h2_target < h2_reachable_min:
            h2_unreachable = h2_reachable_min - h2_target
        else:
            h2_unreachable = 0
        
        # ==================== 时变权重计算 ====================
        # k(t) = k_base * (1 + (t/23)^alpha)，越接近终点权重越大
        time_weight = 1 + (t / 23) ** self.funnel_alpha
        k_funnel = self.k_funnel_base * time_weight
        
        # ==================== 不可达惩罚（平方形式，超出容差部分） ====================
        soc_excess = max(0, soc_unreachable - self.soc_tolerance)
        h2_excess = max(0, h2_unreachable - self.h2_soc_tolerance)
        
        penalty = k_funnel * (soc_excess ** 2 + h2_excess ** 2)
        
        funnel_info = {
            'soc_unreachable': soc_unreachable,
            'h2_unreachable': h2_unreachable,
            'soc_reachable_range': (soc_reachable_min, soc_reachable_max),
            'h2_reachable_range': (h2_reachable_min, h2_reachable_max),
            'time_weight': time_weight,
            'k_funnel': k_funnel
        }
        
        return penalty, funnel_info

    def add_uncertainty(self, data, uncertainty_ratio=0.01):
        """为数据添加不确定性波动（使用env随机数保证可复现）"""
        return [value * (1 + self.np_random.uniform(-uncertainty_ratio, uncertainty_ratio)) 
                for value in data]

    def get_elec_efficiency(self, power_ratio):
        """电解槽变效率特性（模拟极化曲线）"""
        if power_ratio < self.elec_min_load:
            return 0.0
        else:
            return 0.65 * (1 - 0.2 * (1 - power_ratio)**2)

    def get_fc_efficiency(self, power_ratio):
        """燃料电池变效率特性"""
        if power_ratio < self.fc_min_load:
            return 0.0
        else:
            return 0.50 * (1 - 0.15 * (1 - power_ratio)**2)

    def project_action(self, action_cmd, c_soc, h2_soc, t):
        """
        动作投影函数：将网络输出的动作投影到可行域，统一处理所有约束
        参数:
            action_cmd: 网络输出的原始动作 [p_elec, p_bes, p_fc, h2_sell]
            c_soc: 当前电池SOC
            h2_soc: 当前储氢SOC
            t: 当前时刻
        返回:
            action_exec: 实际执行的动作字典
            violation: 各类约束违反量字典
            aux_info: 辅助信息（效率、氢气产量、SOC等）
        """
        violation = {
            'elec_power': 0, 'elec_ramp': 0,
            'fc_power': 0, 'fc_ramp': 0,
            'bes_power': 0, 'bes_soc': 0,
            'h2_sell': 0, 'h2_soc': 0,
            'grid_over': 0
        }
        
        p_elec_cmd, p_bes_cmd, p_fc_cmd, h2_sell_cmd = action_cmd
        
        # ==================== 1. 功率上下限约束 ====================
        p_elec = np.clip(p_elec_cmd, self.min_p_elec, self.max_p_elec)
        p_fc = np.clip(p_fc_cmd, self.min_p_fc, self.max_p_fc)
        p_bes = np.clip(p_bes_cmd, self.min_p_bes, self.max_p_bes)
        h2_sell = np.clip(h2_sell_cmd, 0, self.max_h2_sell)
        
        violation['elec_power'] = abs(p_elec_cmd - p_elec)
        violation['fc_power'] = abs(p_fc_cmd - p_fc)
        violation['bes_power'] = abs(p_bes_cmd - p_bes)
        violation['h2_sell'] = abs(h2_sell_cmd - h2_sell)
        
        # ==================== 2. 爬坡约束 ====================
        elec_ramp_up = self.elec_ramp_up * self.max_p_elec
        elec_ramp_down = self.elec_ramp_down * self.max_p_elec
        p_elec_max = min(self.last_p_elec + elec_ramp_up, self.max_p_elec)
        p_elec_min = max(self.last_p_elec - elec_ramp_down, self.min_p_elec)
        p_elec_before_ramp = p_elec
        p_elec = np.clip(p_elec, p_elec_min, p_elec_max)
        violation['elec_ramp'] = abs(p_elec_before_ramp - p_elec)
        
        fc_ramp_up = self.fc_ramp_up * self.max_p_fc
        fc_ramp_down = self.fc_ramp_down * self.max_p_fc
        p_fc_max = min(self.last_p_fc + fc_ramp_up, self.max_p_fc)
        p_fc_min = max(self.last_p_fc - fc_ramp_down, self.min_p_fc)
        p_fc_before_ramp = p_fc
        p_fc = np.clip(p_fc, p_fc_min, p_fc_max)
        violation['fc_ramp'] = abs(p_fc_before_ramp - p_fc)
        
        # ==================== 3. 死区约束与效率计算 ====================
        elec_load_ratio = p_elec / self.max_p_elec if self.max_p_elec > 0 else 0
        fc_load_ratio = p_fc / self.max_p_fc if self.max_p_fc > 0 else 0
        
        if elec_load_ratio < self.elec_min_load:
            p_elec = 0
            elec_eff = 0
        else:
            elec_eff = self.get_elec_efficiency(elec_load_ratio)
            
        if fc_load_ratio < self.fc_min_load:
            p_fc = 0
            fc_eff = 0
        else:
            fc_eff = self.get_fc_efficiency(fc_load_ratio)
        
        # ==================== 4. 储能SOC约束（clip功率保证能量守恒） ====================
        max_charge_power = (self.max_c_soc - c_soc) * self.q_bes / self.n_ch / self.time_step
        max_discharge_power = (c_soc - self.min_c_soc) * self.q_bes * self.n_dis / self.time_step
        
        p_bes_before_soc = p_bes
        if p_bes < 0:  # 充电
            p_bes = max(p_bes, -max_charge_power)
        else:  # 放电
            p_bes = min(p_bes, max_discharge_power)
        violation['bes_soc'] = abs(p_bes_before_soc - p_bes)
        
        # ==================== 5. 售氢库存约束 ====================
        current_h2_kg = h2_soc * self.h2_storage_cap
        h2_sell_before = h2_sell
        h2_sell = min(h2_sell, current_h2_kg * 0.9)
        violation['h2_sell'] += abs(h2_sell_before - h2_sell)
        
        # ==================== 6. 计算氢气产量和消耗 ====================
        h2_prod = p_elec * elec_eff / self.LHV_H2 * 1000 if elec_eff > 0 else 0
        h2_consume = p_fc / fc_eff / self.LHV_H2 * 1000 if fc_eff > 0 else 0
        
        # ==================== 7. 储氢SOC约束（闭环投影：超上限削减产氢，超下限削减用氢/售氢） ====================
        new_h2_soc_raw = h2_soc + (h2_prod - h2_consume - h2_sell) / self.h2_storage_cap
        
        if new_h2_soc_raw > self.max_h2_soc:
            excess_h2 = (new_h2_soc_raw - self.max_h2_soc) * self.h2_storage_cap
            violation['h2_soc'] = excess_h2
            if h2_prod > 0:
                h2_prod_new = max(0, h2_prod - excess_h2)
                p_elec = h2_prod_new * self.LHV_H2 / 1000 / (elec_eff + 1e-8) if elec_eff > 0 else 0
                p_elec = np.clip(p_elec, self.min_p_elec, self.max_p_elec)
                p_elec = np.clip(p_elec, p_elec_min, p_elec_max)
                elec_load_ratio = p_elec / self.max_p_elec if self.max_p_elec > 0 else 0
                elec_eff = self.get_elec_efficiency(elec_load_ratio) if elec_load_ratio >= self.elec_min_load else 0
                h2_prod = p_elec * elec_eff / self.LHV_H2 * 1000 if elec_eff > 0 else 0
            new_h2_soc_raw = h2_soc + (h2_prod - h2_consume - h2_sell) / self.h2_storage_cap
        elif new_h2_soc_raw < self.min_h2_soc:
            deficit_h2 = (self.min_h2_soc - new_h2_soc_raw) * self.h2_storage_cap
            h2_sell_reduce = min(h2_sell, deficit_h2)
            h2_sell -= h2_sell_reduce
            deficit_h2 -= h2_sell_reduce
            if deficit_h2 > 0 and h2_consume > 0:
                h2_consume_reduce = min(h2_consume, deficit_h2)
                h2_consume -= h2_consume_reduce
                p_fc = h2_consume * fc_eff * self.LHV_H2 / 1000 if fc_eff > 0 else 0
                deficit_h2 -= h2_consume_reduce
            violation['h2_soc'] = deficit_h2 if deficit_h2 > 0 else 0
        
        # ==================== 7.5 日终SOC硬闭合（仅测试时启用，训练时关闭让策略学习闭合） ====================
        if t == 23 and not self.training_mode:
            # 储能SOC闭合：计算当前p_bes执行后的SOC，然后调整p_bes使终端SOC=目标
            # 先计算如果用当前p_bes会得到什么SOC
            if p_bes < 0:
                n_bes_temp = self.n_ch
            else:
                n_bes_temp = 1 / self.n_dis if self.n_dis > 0 else 1
            soc_after_current = c_soc - n_bes_temp * p_bes * self.time_step / self.q_bes
            
            # 计算需要的SOC调整量
            soc_target = self.soc_init
            delta_soc_needed = soc_target - soc_after_current  # 正值需要充电，负值需要放电
            
            if abs(delta_soc_needed) > 0.001:  # 需要调整
                if delta_soc_needed > 0:  # 需要额外充电：p_bes变负
                    p_bes_adjust = -delta_soc_needed * self.q_bes / self.n_ch / self.time_step
                else:  # 需要额外放电：p_bes变正
                    p_bes_adjust = -delta_soc_needed * self.q_bes * self.n_dis / self.time_step
                
                p_bes_new = p_bes + p_bes_adjust
                p_bes_new = np.clip(p_bes_new, self.min_p_bes, self.max_p_bes)
                p_bes_new = np.clip(p_bes_new, -max_charge_power, max_discharge_power)
                p_bes_before_terminal = p_bes
                p_bes = p_bes_new
                violation['bes_power'] += abs(p_bes_before_terminal - p_bes)
            
            # 储氢SOC闭合（通过调整h2_sell或p_elec）
            h2_soc_target = self.h2_soc_init
            new_h2_soc_check = h2_soc + (h2_prod - h2_consume - h2_sell) / self.h2_storage_cap
            delta_h2_soc = h2_soc_target - new_h2_soc_check  # 正值=SOC过低，负值=SOC过高
            
            if delta_h2_soc < -0.001:  # SOC过高，需要增加售氢
                excess_h2 = -delta_h2_soc * self.h2_storage_cap
                h2_sell_before_terminal = h2_sell
                h2_sell = min(h2_sell + excess_h2, self.max_h2_sell)
                h2_sell = min(h2_sell, h2_soc * self.h2_storage_cap * 0.9)
                violation['h2_sell'] += abs(h2_sell - h2_sell_before_terminal)
            # 注：SOC过低时（delta_h2_soc > 0），无法在最后一步通过增加产氢弥补，只能靠训练时funnel引导
        
        # ==================== 8. 功率平衡与并网约束（SOC可行域已并入available计算） ====================
        p_grid = p_load_24h[t] + p_elec - p_pv_24h[t] - p_wt_24h[t] - p_bes - p_fc
        
        p_shortage = 0
        p_curtail = 0
        
        # t==23时保护p_bes不被二次修改（仅测试时启用，训练时让策略学习全程规划）
        protect_bes = (t == 23 and not self.training_mode)
        
        if p_grid > self.max_p_grid:
            delta = p_grid - self.max_p_grid
            violation['grid_over'] = delta
            if not protect_bes:
                available_discharge = min(delta, max_discharge_power - max(0, p_bes), self.max_p_bes - p_bes)
                available_discharge = max(0, available_discharge)
                if available_discharge > 0:
                    p_bes += available_discharge
                p_grid = p_load_24h[t] + p_elec - p_pv_24h[t] - p_wt_24h[t] - p_bes - p_fc
            if p_grid > self.max_p_grid:
                p_shortage = p_grid - self.max_p_grid
                p_grid = self.max_p_grid
            
        elif p_grid < self.min_p_grid:
            delta = self.min_p_grid - p_grid
            violation['grid_over'] = delta
            if not protect_bes:
                available_charge = min(delta, max_charge_power + min(0, p_bes), abs(self.min_p_bes) - abs(min(0, p_bes)))
                available_charge = max(0, available_charge)
                if available_charge > 0:
                    p_bes -= available_charge
                p_grid = p_load_24h[t] + p_elec - p_pv_24h[t] - p_wt_24h[t] - p_bes - p_fc
            if p_grid < self.min_p_grid:
                p_curtail = self.min_p_grid - p_grid
                p_grid = self.min_p_grid
        
        # ==================== 9. 闭环一致性：重新计算SOC（不clip） ====================
        if p_bes < 0:
            n_bes = self.n_ch
        else:
            n_bes = 1 / self.n_dis if self.n_dis > 0 else 1
        new_c_soc = c_soc - n_bes * p_bes * self.time_step / self.q_bes
        
        # 重新计算氢气SOC
        new_h2_soc = h2_soc + (h2_prod - h2_consume - h2_sell) / self.h2_storage_cap
        
        # 构建返回结果
        action_exec = {
            'p_elec': p_elec,
            'p_fc': p_fc,
            'p_bes': p_bes,
            'p_grid': p_grid,
            'h2_sell': h2_sell
        }
        
        aux_info = {
            'elec_eff': elec_eff,
            'fc_eff': fc_eff,
            'h2_prod': h2_prod,
            'h2_consume': h2_consume,
            'new_c_soc': new_c_soc,
            'new_h2_soc': new_h2_soc,
            'p_shortage': p_shortage,
            'p_curtail': p_curtail,
            'max_charge_power': max_charge_power,
            'max_discharge_power': max_discharge_power
        }
        
        return action_exec, violation, aux_info

    def step(self, action):
        """
        执行一步动作（标准Gym接口）
        参数:
            action: 神经网络输出的动作 [p_elec, p_bes, p_fc, h2_sell]
        返回:
            obs: 下一状态
            reward: 奖励值
            done: 是否结束
            info: 调试信息字典
        """
        t = self.t
        c_soc = float(self.state[3])
        h2_soc = float(self.state[4])
        
        # 保存原始命令
        action_cmd = action.copy()
        
        # ==================== 调用投影函数获取执行动作 ====================
        action_exec, violation, aux_info = self.project_action(action_cmd, c_soc, h2_soc, t)
        
        # 提取执行动作
        p_elec = action_exec['p_elec']
        p_fc = action_exec['p_fc']
        p_bes = action_exec['p_bes']
        p_grid = action_exec['p_grid']
        h2_sell = action_exec['h2_sell']
        
        # 提取辅助信息
        elec_eff = aux_info['elec_eff']
        fc_eff = aux_info['fc_eff']
        h2_prod = aux_info['h2_prod']
        h2_consume = aux_info['h2_consume']
        new_c_soc = aux_info['new_c_soc']
        new_h2_soc = aux_info['new_h2_soc']
        p_shortage = aux_info['p_shortage']
        p_curtail = aux_info['p_curtail']
        
        # 统一变量名（兼容后续代码）
        action_p_elec = p_elec
        action_p_fc = p_fc
        action_p_bes = p_bes
        action_p_grid = p_grid
        real_h2_sell = h2_sell

        # ==================== 计算惩罚项 ====================
        # cmd-exec差值惩罚（归一化各维度）
        k_violation = 100
        exec_vec = np.array([p_elec, p_bes, p_fc, h2_sell])
        scale = (self.action_space.high - self.action_space.low).astype(np.float32)
        cmd_exec_diff = (np.abs(action_cmd - exec_vec) / (scale + 1e-6)).sum()
        punishment_cmd_exec = k_violation * cmd_exec_diff
        
        # 分项惩罚（用于调试）
        punishment_p_elec = k_violation * (violation['elec_power'] + violation['elec_ramp'])
        punishment_p_fc = k_violation * (violation['fc_power'] + violation['fc_ramp'])
        punishment_p_bes = k_violation * (violation['bes_power'] + violation['bes_soc'])
        punishment_h2_soc = k_violation * violation['h2_soc']
        punishment_h2_sell = k_violation * violation['h2_sell']
        punishment_elec_ramp = 0
        punishment_fc_ramp = 0
        
        # 缺电/弃电惩罚（使用能量MWh，VOLL方式）
        voll = price_ele_buy[t] * 50  # 缺电价值 = 当前购电价 × 50倍
        curtail_price = price_ele_sell[t] * 2  # 弃电损失 = 当前售电价 × 2倍
        E_shortage = p_shortage * self.time_step  # 缺电能量 MWh
        E_curtail = p_curtail * self.time_step    # 弃电能量 MWh
        punishment_shortage = voll * E_shortage
        punishment_curtail = curtail_price * E_curtail
        
        # ==================== 可达性漏斗惩罚（替代固定权重shaping） ====================
        funnel_penalty, funnel_info = self.compute_reachability_penalty(new_c_soc, new_h2_soc, t)
        punishment_soc_closure = funnel_penalty

        #########################################  奖励回报函数  ########################################
        # 判断是否为最后一步（用于terminal惩罚）
        done_next = (self.t + 1 >= 24)
        
        # 计算净负荷（用于弱偏好判断）
        net_load = p_load_24h[t] - p_pv_24h[t] - p_wt_24h[t]
        
        # 购电成本（单位：元）
        if action_p_grid >= 0:
            cost_grid_buy = price_ele_buy[t] * action_p_grid * self.time_step
            revenue_grid_sell = 0
        else:
            cost_grid_buy = 0
            revenue_grid_sell = price_ele_sell[t] * abs(action_p_grid) * self.time_step

        # 售氢收益（单位：元）
        revenue_h2_sell = price_h2 * real_h2_sell

        # 碳交易收益（单位：元）
        carbon_h2 = c_H2 * real_h2_sell  # kg_CO2
        carbon_fc = c_CO2 * 1000 * action_p_fc * self.time_step  # kg_CO2
        if action_p_grid < 0:
            carbon_grid = c_e * 1000 * abs(action_p_grid) * self.time_step  # kg_CO2
            revenue_carbon = P_CO2 * (carbon_grid + carbon_h2 + carbon_fc)
        else:
            revenue_carbon = P_CO2 * (carbon_h2 + carbon_fc)

        # 运维成本（单位：元）
        cost_elec = 0.015 * action_p_elec * self.time_step * 1000
        cost_fc = 0.020 * action_p_fc * self.time_step * 1000

        # 退化成本（单位：元）
        cost_p_bes = self.c_bat_deg * abs(action_p_bes) * self.time_step * 1000
        cost_elec_deg = self.c_elec_deg * action_p_elec * self.time_step * 1000
        cost_fc_deg = self.c_fc_deg * action_p_fc * self.time_step * 1000

        # 启停成本（单位：元）
        elec_startup_cost = 0
        fc_startup_cost = 0
        if action_p_elec > 0 and not self.elec_was_on:
            elec_startup_cost = self.elec_startup_cost
        if action_p_fc > 0 and not self.fc_was_on:
            fc_startup_cost = self.fc_startup_cost

        # 更新启停状态
        self.elec_was_on = (action_p_elec > 0)
        self.fc_was_on = (action_p_fc > 0)

        # 更新上一时刻功率记录
        self.last_p_elec = action_p_elec
        self.last_p_fc = action_p_fc

        # ==================== 弱偏好引导（替代硬互斥约束） ====================
        # 分时段策略引导（不乘1000）
        k_time_guide = 500  # 分时引导系数（元/MWh）
        if 12 <= t <= 18:  # 峰时段
            cost_peak_elec = action_p_elec * k_time_guide  # 峰时制氢小惩罚
        else:
            cost_peak_elec = 0

        if t < 7 or t >= 23:  # 谷时段
            bonus_valley_elec = action_p_elec * k_time_guide  # 谷时制氢小奖励
        else:
            bonus_valley_elec = 0
        
        # 弱偏好：余电时鼓励制氢，缺电时鼓励燃料电池（不乘1000）
        k_preference = 200  # 弱偏好系数（元/MWh）
        if net_load < 0:  # 余电（新能源出力 > 负荷）
            bonus_elec_surplus = action_p_elec * k_preference
            penalty_fc_surplus = action_p_fc * k_preference * 0.5  # 余电时FC小惩罚
        else:  # 缺电（负荷 > 新能源出力）
            bonus_elec_surplus = 0
            penalty_fc_surplus = 0

        # ==================== 真实财务账（用于画图） ====================
        real_revenue = revenue_grid_sell + revenue_h2_sell + revenue_carbon
        real_cost = cost_grid_buy + cost_p_bes + cost_elec + cost_fc + elec_startup_cost + fc_startup_cost + cost_elec_deg + cost_fc_deg
        profit_t = real_revenue - real_cost  # 单位：元

        # 真实收益（用于画图，除以100转换单位）
        money_cost = profit_t / 100

        # ==================== RL奖励计算（归一化） ====================
        # 奖励尺度S，使reward大致在[-10, 10]范围
        REWARD_SCALE = 5000.0
        
        # 利润项（归一化）
        reward_profit = profit_t / REWARD_SCALE
        
        # 引导项（归一化）
        guide_bonus = (bonus_valley_elec + bonus_elec_surplus - cost_peak_elec - penalty_fc_surplus) / REWARD_SCALE
        
        # 惩罚项（归一化，分项惩罚仅用于log，只保留cmd_exec惩罚避免重复）
        total_punishment = (punishment_shortage + punishment_curtail + punishment_soc_closure +
                           punishment_cmd_exec)
        reward_punishment = -total_punishment / REWARD_SCALE
        
        # 总奖励
        reward = reward_profit + guide_bonus + reward_punishment
        
        # 日终SOC闭合惩罚
        if done_next:
            reward -= self.k_terminal_soc * abs(new_c_soc - 0.5) / REWARD_SCALE
            reward -= self.k_terminal_h2 * abs(new_h2_soc - 0.5) / REWARD_SCALE

        ################################  执行一步动作后的状态更新  ################################
        # 更新内部时间
        self.t += 1
        done = done_next
        
        # 获取下一时刻的负荷和风光数据（处理边界）
        if done:
            t_next = 0  # episode结束，下一状态用0时刻（不会被使用）
        else:
            t_next = self.t
        
        p_load_next = p_load_24h[t_next]
        p_wt_next = p_wt_24h[t_next]
        p_pv_next = p_pv_24h[t_next]

        # 构建14维状态向量
        sin_t_next = np.sin(2 * np.pi * t_next / 24)
        cos_t_next = np.cos(2 * np.pi * t_next / 24)
        price_buy_norm = price_ele_buy[t_next] / self.price_buy_max
        price_sell_norm = price_ele_sell[t_next] / self.price_sell_max
        last_p_elec_norm = self.last_p_elec / self.max_p_elec if self.max_p_elec > 0 else 0
        last_p_fc_norm = self.last_p_fc / self.max_p_fc if self.max_p_fc > 0 else 0
        net_load_next = p_load_next - p_pv_next - p_wt_next
        
        self.state = np.array([
            np.clip(p_load_next / self.P_SCALE, 0, 1), 
            np.clip(p_wt_next / self.P_SCALE, 0, 1), 
            np.clip(p_pv_next / self.P_SCALE, 0, 1), 
            new_c_soc, new_h2_soc,
            sin_t_next, cos_t_next, price_buy_norm, price_sell_norm,
            last_p_elec_norm, last_p_fc_norm,
            1.0 if self.elec_was_on else 0.0,
            1.0 if self.fc_was_on else 0.0,
            np.clip(net_load_next / self.NET_SCALE, -1, 1)
        ], dtype=np.float32)
        
        # 构建info字典（用于调试和记录）
        info = {
            'action_exec': np.array([action_p_grid, action_p_elec, action_p_bes, action_p_fc]),
            'action_cmd': action_cmd,
            'h2_prod': h2_prod,
            'h2_consume': h2_consume,
            'h2_sell': real_h2_sell,
            'p_shortage': p_shortage,
            'p_curtail': p_curtail,
            'violation': violation,
            'cost_grid_buy': cost_grid_buy,
            'revenue_grid_sell': revenue_grid_sell,
            'revenue_h2_sell': revenue_h2_sell,
            'revenue_carbon': revenue_carbon,
            'cost_p_bes': cost_p_bes,
            'cost_elec': cost_elec,
            'cost_fc': cost_fc,
            'money_cost': money_cost,
            'punishment_p_elec': punishment_p_elec,
            'punishment_p_bes': punishment_p_bes,
            'punishment_p_fc': punishment_p_fc,
            'punishment_h2_soc': punishment_h2_soc,
            'punishment_h2_sell': punishment_h2_sell,
            'punishment_shortage': punishment_shortage,
            'punishment_curtail': punishment_curtail,
            'c_soc': new_c_soc,
            'h2_soc': new_h2_soc,
            't': t,  # 当前时刻（执行动作时的时刻）
            'funnel_penalty': funnel_penalty,
            'funnel_info': funnel_info
        }

        return self.state, reward, done, info

    def reset(self):
        """
        环境重置函数: 每个回合开始时重新生成带有不确定性的负荷和风光数据
        返回:
            state: 初始状态
        """
        global p_load_24h, p_wt_24h, p_pv_24h
        p_load_24h = [i * 1 for i in self.add_uncertainty(p_load_24h0)]
        p_wt_24h = [i * 1 for i in self.add_uncertainty(p_wt_24h0)]
        p_pv_24h = [i * 1 for i in self.add_uncertainty(p_pv_24h0)]
        
        # 初始化内部时间
        self.t = 0
        
        p_load = p_load_24h[self.t]
        p_wt = p_wt_24h[self.t]
        p_pv = p_pv_24h[self.t]

        self.c_soc = 0.5
        self.h2_soc = 0.5
        self.soc_init = self.c_soc
        self.h2_soc_init = self.h2_soc
        self.elec_was_on = False
        self.fc_was_on = False
        self.last_p_elec = 0
        self.last_p_fc = 0
        
        # 构建14维状态向量
        sin_t = np.sin(2 * np.pi * self.t / 24)
        cos_t = np.cos(2 * np.pi * self.t / 24)
        price_buy_norm = price_ele_buy[self.t] / self.price_buy_max
        price_sell_norm = price_ele_sell[self.t] / self.price_sell_max
        last_p_elec_norm = self.last_p_elec / self.max_p_elec if self.max_p_elec > 0 else 0
        last_p_fc_norm = self.last_p_fc / self.max_p_fc if self.max_p_fc > 0 else 0
        net_load = p_load - p_pv - p_wt
        
        self.state = np.array([
            np.clip(p_load / self.P_SCALE, 0, 1), 
            np.clip(p_wt / self.P_SCALE, 0, 1), 
            np.clip(p_pv / self.P_SCALE, 0, 1), 
            self.c_soc, self.h2_soc,
            sin_t, cos_t, price_buy_norm, price_sell_norm,
            last_p_elec_norm, last_p_fc_norm,
            0.0, 0.0,  # elec_on, fc_on
            np.clip(net_load / self.NET_SCALE, -1, 1)
        ], dtype=np.float32)
        return self.state

#####################################  设置命令行参数  ##############################
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)  # 训练模式标志
parser.add_argument('--test', dest='test', action='store_true', default=True)   # 测试模式标志
args = parser.parse_args()

#########################################  PPO网络与算法  #################################
class Actor(nn.Module):
    """
    Actor网络类: 用于生成动作的策略网络
    """
    def __init__(self, state_dim, action_dim, action_bound_high, action_bound_low):
        """
        初始化Actor网络
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            action_bound_high: 动作上界
            action_bound_low: 动作下界
        """
        super(Actor, self).__init__()
        self.action_bound_high = torch.FloatTensor(action_bound_high)  # 动作上限
        self.action_bound_low = torch.FloatTensor(action_bound_low)   # 动作下限
        
        # 修改网络结构：增加一层256节点的全连接层
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)  # 新增层
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, x):
        """
        前向传播函数
        参数:
            x: 输入状态
        返回:
            mu: 动作均值
            std: 动作标准差
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # 新增层的激活
        mu = self.mu_head(x)  # 输出原始mu，tanh-squash由PPO统一负责
        std = torch.exp(self.log_std_head)
        return mu, std

class Critic(nn.Module):
    """
    Critic网络类: 用于评估状态价值的价值网络
    """
    def __init__(self, state_dim):
        """
        初始化Critic网络
        参数:
            state_dim: 状态维度
        """
        super(Critic, self).__init__()
        # 修改网络结构：增加一层256节点的全连接层
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)  # 新增层
        self.fc4 = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        前向传播函数
        参数:
            x: 输入状态
        返回:
            value: 状态价值估计
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # 新增层的激活
        value = self.fc4(x)
        return value

class PPO:
    """
    PPO算法主类: 实现PPO算法的核心功能
    改进：GAE(λ) + Entropy Bonus + Advantage Normalization + Tanh-Squash
    """
    def __init__(self, state_dim, action_dim, action_bound_high, action_bound_low):
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # 动作边界
        self.action_bound_high = torch.FloatTensor(action_bound_high).to(self.device)
        self.action_bound_low = torch.FloatTensor(action_bound_low).to(self.device)
        self.action_dim = action_dim
        
        # 创建Actor和Critic网络
        self.actor = Actor(state_dim, action_dim, action_bound_high, action_bound_low).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        self.actor.action_bound_high = self.actor.action_bound_high.to(self.device)
        self.actor.action_bound_low = self.actor.action_bound_low.to(self.device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_C)
        
        # 经验缓冲区（扩展为存储更多信息）
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []
        
        # GAE参数
        self.gae_lambda = 0.95
        
        # Entropy bonus系数
        self.entropy_coef = 0.001

    def store_transition(self, state, action, reward, done, value, log_prob):
        """存储经验到缓冲区（扩展版）"""
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.value_buffer.append(value)
        self.log_prob_buffer.append(log_prob)

    def get_action(self, state, greedy=False):
        """
        根据状态选择动作（Tanh-Squash版本）
        返回: action, value, log_prob
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取网络输出的均值和标准差
            mu, std = self.actor(state_tensor)
            value = self.critic(state_tensor).item()
            
            if greedy:
                u = mu
                action_tanh = torch.tanh(u)
                action = action_tanh * (self.action_bound_high - self.action_bound_low) / 2 + \
                        (self.action_bound_high + self.action_bound_low) / 2
                log_prob = 0.0
            else:
                # 采样未压缩的动作
                dist = Normal(mu, std)
                u = dist.rsample()  # 重参数化采样
                
                # Tanh压缩到[-1, 1]
                action_tanh = torch.tanh(u)
                
                # 缩放到动作边界
                action = action_tanh * (self.action_bound_high - self.action_bound_low) / 2 + \
                        (self.action_bound_high + self.action_bound_low) / 2
                
                # 计算修正后的log_prob（Tanh的Jacobian修正）
                log_prob_raw = dist.log_prob(u).sum(dim=1)
                # Jacobian修正: log|det(d tanh(u)/du)| = sum(log(1 - tanh^2(u)))
                log_prob_correction = torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=1)
                log_prob = (log_prob_raw - log_prob_correction).item()
        
        return action.cpu().detach().numpy().flatten(), value, log_prob

    def compute_gae(self, last_value, done):
        """
        计算GAE(λ) advantage
        """
        rewards = np.array(self.reward_buffer)
        values = np.array(self.value_buffer + [last_value])
        dones = np.array(self.done_buffer)
        
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * values[t + 1] * mask - values[t]
            gae = delta + GAMMA * self.gae_lambda * mask * gae
            advantages[t] = gae
        
        returns = advantages + np.array(self.value_buffer)
        return advantages, returns

    def update(self, last_state, done):
        """
        更新策略网络和价值网络（GAE + Entropy + Advantage Norm）
        """
        if len(self.state_buffer) == 0:
            return
        
        # 计算最后状态的价值
        with torch.no_grad():
            last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            last_value = self.critic(last_state_tensor).item() if not done else 0
        
        # 计算GAE
        advantages, returns = self.compute_gae(last_value, done)
        
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.state_buffer)).to(self.device)
        actions = torch.FloatTensor(np.array(self.action_buffer)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_prob_buffer)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        
        # Advantage normalization
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO更新
        for _ in range(ACTOR_UPDATE_STEPS):
            # 重新计算log_prob和entropy
            mu, std = self.actor(states)
            dist = Normal(mu, std)
            
            # 反向计算u（从action反推）
            action_normalized = (actions - (self.action_bound_high + self.action_bound_low) / 2) / \
                               ((self.action_bound_high - self.action_bound_low) / 2 + 1e-8)
            action_normalized = torch.clamp(action_normalized, -0.999, 0.999)
            u = torch.atanh(action_normalized)
            
            log_prob_raw = dist.log_prob(u).sum(dim=1)
            log_prob_correction = torch.log(1 - action_normalized.pow(2) + 1e-6).sum(dim=1)
            new_log_probs = log_prob_raw - log_prob_correction
            
            # 计算entropy
            entropy = dist.entropy().sum(dim=1).mean()
            
            # 计算ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO-Clip目标
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
        
        # 更新Critic
        for _ in range(CRITIC_UPDATE_STEPS):
            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns_tensor)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # 清空缓冲区
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.value_buffer.clear()
        self.log_prob_buffer.clear()

    def save(self):
        torch.save(self.actor.cpu().state_dict(), 'ppo_actor.pth')
        torch.save(self.critic.cpu().state_dict(), 'ppo_critic.pth')
        self.actor.to(self.device)
        self.critic.to(self.device)
        
    def load(self):
        """
        加载模型参数
        """
        self.actor.load_state_dict(torch.load('ppo_actor.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load('ppo_critic.pth', map_location=self.device))

if __name__ == '__main__':
    env = Environment()  # 创建环境实例
    
    # 设置随机种子,确保结果可复现
    torch.manual_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 获取状态和动作的维度信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound_high = env.action_space.high
    action_bound_low = env.action_space.low
    
    # 创建PPO智能体
    agent = PPO(state_dim, action_dim, action_bound_high, action_bound_low)
    
    if args.train:  # 训练模式
        # 初始化记录列表
        all_episode_reward = []  # 记录每轮的EMA奖励
        raw_episode_rewards = []  # 记录每轮的原始奖励
        money_list = []         # 记录每轮的总成本
        
        t0 = time.time()  # 记录训练开始时间
        
        # 开始训练循环
        for episode in range(TRAIN_EPISODES):
            state = env.reset()  # 重置环境
            episode_reward = 0
            money_day = 0
            
            # 单轮训练循环
            for step in range(MAX_STEPS):
                # 获取动作、价值和log_prob
                action, value, log_prob = agent.get_action(state)
                state_, reward, done, info = env.step(action)
                
                money_day += info['money_cost']
                
                # 存储经验（扩展版）
                agent.store_transition(state, action, reward, done, value, log_prob)
                state = state_
                episode_reward += reward
                
                if done:
                    break
            
            # Episode结束，更新网络
            agent.update(state_, done)
            
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0)
            )
            
            raw_episode_rewards.append(episode_reward)
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
        
        # 保存训练好的模型
        agent.save()
        
        # 绘图展示训练结果
        plt.figure(figsize=(12, 6))
        plt.plot(raw_episode_rewards, alpha=0.3, label='Raw Reward')
        plt.plot(all_episode_reward, linewidth=2, label='EMA Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('Training Progress')
        if not os.path.exists('image'):
            os.makedirs('image')
        reward_file = "./image/PPO.png"
        if os.path.exists(reward_file):
            os.remove(reward_file)
        plt.savefig(reward_file)
        plt.close()

        # 保存训练数据到CSV文件
        all_episode_reward = pd.DataFrame(all_episode_reward)
        all_episode_reward.to_csv('all_episode_reward.csv')

    if args.test:  # 测试模式
        agent.load()  # 加载训练好的模型
        env.training_mode = False  # 测试时启用硬闭合作为safety fallback
        
        # 初始化测试数据记录列表
        action_p_elec2 = []  
        action_p_bes2 = [] 
        action_p_fc2 = []   
        action_p_grid2 = [] 
        h2_prod_list = []   
        h2_consume_list = []
        h2_sell_list = []
        p_wt_list = []
        p_pv_list = []
        p_load_list = []
        
        # 记录惩罚和成本
        punishment_p_elec_list = []
        punishment_p_bes_list = []
        punishment_p_fc_list = []
        punishment_h2_soc_list = []
        punishment_h2_sell_list = []
        cost_grid_buy_list = []
        revenue_grid_sell_list = []
        revenue_h2_sell_list = []
        revenue_carbon_list = []
        cost_p_bes_list = []
        cost_elec_list = []
        cost_fc_list = []
        money_cost_list = []
        c_soc_list = []
        c_soc_pre_list = []   # step前SOC
        c_soc_post_list = []  # step后SOC
        h2_soc_list = []
        h2_soc_pre_list = []  # step前H2 SOC
        h2_soc_post_list = [] # step后H2 SOC
        p_shortage_list = []
        p_curtail_list = []
        
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            money_day = 0
            
            for step in range(TEST_MAX_STEPS):
                # 记录当前状态（直接从全局序列取真实MW值，避免clip截断）
                t_now = env.t
                p_load_list.append(p_load_24h[t_now])
                p_wt_list.append(p_wt_24h[t_now])
                p_pv_list.append(p_pv_24h[t_now])
                c_soc_list.append(state[3])
                c_soc_pre_list.append(state[3])  # step前SOC
                h2_soc_list.append(state[4])
                h2_soc_pre_list.append(state[4]) # step前H2 SOC
                
                # 测试时使用greedy模式，只需要action
                action, _, _ = agent.get_action(state, greedy=True)
                state_, reward, done, info = env.step(action)
                
                # 记录step后SOC
                c_soc_post_list.append(info['c_soc'])
                h2_soc_post_list.append(info['h2_soc'])
                
                # 从info中提取数据
                action_exec = info['action_exec']
                action_p_grid2.append(action_exec[0])
                action_p_elec2.append(action_exec[1])
                action_p_bes2.append(action_exec[2])
                action_p_fc2.append(action_exec[3])
                h2_prod_list.append(info['h2_prod'])
                h2_consume_list.append(info['h2_consume'])
                h2_sell_list.append(info['h2_sell'])
                p_shortage_list.append(info['p_shortage'])
                p_curtail_list.append(info['p_curtail'])
                
                # 记录惩罚和成本数据
                punishment_p_elec_list.append(info['punishment_p_elec'])
                punishment_p_bes_list.append(info['punishment_p_bes'])
                punishment_p_fc_list.append(info['punishment_p_fc'])
                punishment_h2_soc_list.append(info['punishment_h2_soc'])
                punishment_h2_sell_list.append(info['punishment_h2_sell'])
                cost_grid_buy_list.append(info['cost_grid_buy'])
                revenue_grid_sell_list.append(info['revenue_grid_sell'])
                revenue_h2_sell_list.append(info['revenue_h2_sell'])
                revenue_carbon_list.append(info['revenue_carbon'])
                cost_p_bes_list.append(info['cost_p_bes'])
                cost_elec_list.append(info['cost_elec'])
                cost_fc_list.append(info['cost_fc'])
                money_cost_list.append(info['money_cost'])
                money_day += info['money_cost']
                
                state = state_
                episode_reward += reward
                
                if done:
                    # 打印终端SOC闭合诊断
                    print(f'=== SOC闭合诊断 ===')
                    print(f'电池SOC: 初始={env.soc_init:.4f}, 终端={info["c_soc"]:.4f}, 偏差={info["c_soc"]-env.soc_init:.4f}')
                    print(f'储氢SOC: 初始={env.h2_soc_init:.4f}, 终端={info["h2_soc"]:.4f}, 偏差={info["h2_soc"]-env.h2_soc_init:.4f}')
                    break
            
            print('Test  | Episode: {}/{}  | Episode Reward: {:.4f}'.format(
                episode + 1, TEST_EPISODES, episode_reward))
        
        # 绘制测试结果图表
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        mpl.rcParams["axes.unicode_minus"] = False
        x = np.linspace(1, 24, 24)
        bar_width = 0.35
        
        # 电能展示图1：电网交换和电解槽
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.bar(x, action_p_elec2, bar_width, align="center", label="p_elec")
        ax.bar(x + bar_width, action_p_grid2, bar_width, align="center", label="p_grid")
        ax.legend()
        ax.set_title("电能展示-电解槽与电网")
        fig.tight_layout()
        ele_one_file = "./image/ele_one.png"
        if os.path.exists(ele_one_file):
            os.remove(ele_one_file)
        fig.savefig(ele_one_file)
        plt.close(fig)
        
        # 电能展示图2：储能和燃料电池
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.bar(x, action_p_fc2, bar_width, align="center", label="p_fc")
        ax.bar(x + bar_width, action_p_bes2, bar_width, align="center", label="p_bes")
        ax.legend()
        ax.set_title("电能展示-燃料电池与储能")
        fig.tight_layout()
        ele_two_file = "./image/ele_two.png"
        if os.path.exists(ele_two_file):
            os.remove(ele_two_file)
        fig.savefig(ele_two_file)
        plt.close(fig)
        
        # 氢能展示图
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.bar(x, h2_prod_list, bar_width, label="h2_prod")
        ax.bar(x + bar_width, h2_consume_list, bar_width, label="h2_consume")
        ax.bar(x + 2*bar_width, h2_sell_list, bar_width, label="h2_sell")
        ax.legend()
        ax.set_title("氢能展示-制氢、用氢与售氢")
        fig.tight_layout()
        heat_file = "./image/hydrogen.png"
        if os.path.exists(heat_file):
            os.remove(heat_file)
        fig.savefig(heat_file)
        plt.close(fig)
        
        # 功率调度堆叠图
        fig, ax = plt.subplots(figsize=(20, 10))
        
        p_wt_arr = np.array(p_wt_list)
        p_pv_arr = np.array(p_pv_list)
        p_fc_arr = np.array(action_p_fc2)
        p_bes_arr = np.array(action_p_bes2)
        p_grid_arr = np.array(action_p_grid2)
        p_elec_arr = np.array(action_p_elec2)
        p_load_arr = np.array(p_load_list)
        
        p_bes_dis = np.maximum(p_bes_arr, 0)
        p_bes_ch = np.minimum(p_bes_arr, 0)
        p_grid_buy = np.maximum(p_grid_arr, 0)
        p_grid_sell = np.minimum(p_grid_arr, 0)
        
        # 科研学术简约配色（正向堆叠：光伏→风电→燃料电池→储能放电→购电）
        ax.bar(x, p_pv_arr, width=0.8, label='光伏', color='#F4A460')
        ax.bar(x, p_wt_arr, width=0.8, bottom=p_pv_arr, label='风电', color='#4682B4')
        ax.bar(x, p_fc_arr, width=0.8, bottom=p_pv_arr+p_wt_arr, label='燃料电池', color='#6B8E23')
        ax.bar(x, p_bes_dis, width=0.8, bottom=p_pv_arr+p_wt_arr+p_fc_arr, label='储能放电', color='#9370DB')
        ax.bar(x, p_grid_buy, width=0.8, bottom=p_pv_arr+p_wt_arr+p_fc_arr+p_bes_dis, label='购电', color='#CD5C5C')
        
        # 负向堆叠：电解槽→储能充电→售电
        ax.bar(x, -p_elec_arr, width=0.8, label='电解槽', color='#20B2AA')
        ax.bar(x, p_bes_ch, width=0.8, bottom=-p_elec_arr, label='储能充电', color='#8B4513')
        ax.bar(x, p_grid_sell, width=0.8, bottom=-p_elec_arr+p_bes_ch, label='售电', color='#708090')
        
        ax.plot(x, p_load_arr, 'k-o', linewidth=2, markersize=6, label='电负荷')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('时间 (h)')
        ax.set_ylabel('功率 (MW)')
        ax.set_title('电力系统功率调度图')
        ax.legend(loc='upper right')
        ax.set_xticks(x)
        fig.tight_layout()
        dispatch_file = "./image/power_dispatch.png"
        if os.path.exists(dispatch_file):
            os.remove(dispatch_file)
        fig.savefig(dispatch_file)
        plt.close(fig)
        
        # SOC变化图（双曲线诊断：step前/后SOC + 终端偏差）
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 子图1：电池SOC双曲线
        ax1 = axes[0]
        x_pre = np.arange(1, 25)  # step前：时刻1-24开始时
        x_post = np.arange(1, 25) + 0.5  # step后：时刻1-24结束时（偏移0.5便于区分）
        ax1.plot(x_pre, c_soc_pre_list, 'b-o', linewidth=2, markersize=5, label='电池SOC(step前)')
        ax1.plot(x_post, c_soc_post_list, 'r-s', linewidth=2, markersize=5, label='电池SOC(step后)')
        ax1.axhline(y=env.soc_init, color='gray', linestyle='--', linewidth=1.5, label=f'目标SOC={env.soc_init:.2f}')
        # 标注终端偏差
        terminal_dev = c_soc_post_list[-1] - env.soc_init
        ax1.annotate(f'终端偏差={terminal_dev:.4f}', xy=(24.5, c_soc_post_list[-1]), 
                     xytext=(22, c_soc_post_list[-1]+0.1), fontsize=10, color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))
        ax1.set_xlabel('时间 (h)')
        ax1.set_ylabel('电池SOC')
        ax1.set_ylim([0, 1])
        ax1.set_xticks(np.arange(1, 25))
        ax1.legend(loc='upper right')
        ax1.set_title('电池SOC闭合诊断')
        ax1.grid(True, alpha=0.3)
        
        # 子图2：储氢SOC双曲线
        ax2 = axes[1]
        ax2.plot(x_pre, h2_soc_pre_list, 'g-o', linewidth=2, markersize=5, label='储氢SOC(step前)')
        ax2.plot(x_post, h2_soc_post_list, 'm-s', linewidth=2, markersize=5, label='储氢SOC(step后)')
        ax2.axhline(y=env.h2_soc_init, color='gray', linestyle='--', linewidth=1.5, label=f'目标SOC={env.h2_soc_init:.2f}')
        # 标注终端偏差
        h2_terminal_dev = h2_soc_post_list[-1] - env.h2_soc_init
        ax2.annotate(f'终端偏差={h2_terminal_dev:.4f}', xy=(24.5, h2_soc_post_list[-1]), 
                     xytext=(22, h2_soc_post_list[-1]+0.1), fontsize=10, color='magenta',
                     arrowprops=dict(arrowstyle='->', color='magenta'))
        ax2.set_xlabel('时间 (h)')
        ax2.set_ylabel('储氢SOC')
        ax2.set_ylim([0, 1])
        ax2.set_xticks(np.arange(1, 25))
        ax2.legend(loc='upper right')
        ax2.set_title('储氢SOC闭合诊断')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        soc_file = "./image/soc_change.png"
        if os.path.exists(soc_file):
            os.remove(soc_file)
        fig.savefig(soc_file)
        plt.close(fig)
        
        # 保存测试数据到CSV文件
        action_p_elec2 = pd.DataFrame(action_p_elec2)
        action_p_elec2.to_csv('action_p_elec_test.csv')
        action_p_bes2 = pd.DataFrame(action_p_bes2)
        action_p_bes2.to_csv('action_p_bes_test.csv')
        action_p_fc2 = pd.DataFrame(action_p_fc2)
        action_p_fc2.to_csv('action_p_fc_test.csv')
        action_p_grid2 = pd.DataFrame(action_p_grid2)
        action_p_grid2.to_csv('action_p_grid_test.csv')
        h2_prod_list = pd.DataFrame(h2_prod_list)
        h2_prod_list.to_csv('h2_prod_test.csv')
        h2_consume_list = pd.DataFrame(h2_consume_list)
        h2_consume_list.to_csv('h2_consume_test.csv')
        h2_sell_list = pd.DataFrame(h2_sell_list)
        h2_sell_list.to_csv('h2_sell_test.csv')
        
        # 保存缺电和弃电数据
        p_shortage_df = pd.DataFrame(p_shortage_list)
        p_shortage_df.to_csv('p_shortage.csv')
        p_curtail_df = pd.DataFrame(p_curtail_list)
        p_curtail_df.to_csv('p_curtail.csv')
        
        # 保存成本和收益数据
        cost_grid_buy_list = pd.DataFrame(cost_grid_buy_list)
        cost_grid_buy_list.to_csv('cost_grid_buy.csv')
        revenue_grid_sell_list = pd.DataFrame(revenue_grid_sell_list)
        revenue_grid_sell_list.to_csv('revenue_grid_sell.csv')
        revenue_h2_sell_list = pd.DataFrame(revenue_h2_sell_list)
        revenue_h2_sell_list.to_csv('revenue_h2_sell.csv')
        revenue_carbon_list = pd.DataFrame(revenue_carbon_list)
        revenue_carbon_list.to_csv('revenue_carbon.csv')
        cost_p_bes_list = pd.DataFrame(cost_p_bes_list)
        cost_p_bes_list.to_csv('cost_p_bes.csv')
        cost_elec_list = pd.DataFrame(cost_elec_list)
        cost_elec_list.to_csv('cost_elec.csv')
        cost_fc_list = pd.DataFrame(cost_fc_list)
        cost_fc_list.to_csv('cost_fc.csv')
        money_cost_list = pd.DataFrame(money_cost_list)
        money_cost_list.to_csv('money_cost.csv')

        # 保存调度结果到MD文件
        with open('dispatch_results.md', 'w', encoding='utf-8') as f:
            f.write('# 调度结果汇总\n\n')
            f.write('## 一、设备出力数据 (MW/kg)\n\n')
            f.write('| 时刻 | 电负荷 | 风电 | 光伏 | 电网 | 电解槽 | 储能 | 燃料电池 | 制氢 | 用氢 | 售氢 | 缺电 | 弃电 |\n')
            f.write('|------|--------|------|------|------|--------|------|----------|------|------|------|------|------|\n')
            for i in range(24):
                f.write(f'| {i+1} | {p_load_list[i]:.2f} | {p_wt_list[i]:.2f} | {p_pv_list[i]:.2f} | {action_p_grid2.iloc[i,0]:.2f} | {action_p_elec2.iloc[i,0]:.2f} | {action_p_bes2.iloc[i,0]:.2f} | {action_p_fc2.iloc[i,0]:.2f} | {h2_prod_list.iloc[i,0]:.2f} | {h2_consume_list.iloc[i,0]:.2f} | {h2_sell_list.iloc[i,0]:.2f} | {p_shortage_df.iloc[i,0]:.2f} | {p_curtail_df.iloc[i,0]:.2f} |\n')
            
            f.write('\n## 二、成本与收益数据 (元)\n\n')
            f.write('| 时刻 | 购电成本 | 售电收益 | 售氢收益 | 碳交易收益 | 储能成本 | 电解槽成本 | 燃料电池成本 | 净收益(元/100) |\n')
            f.write('|------|----------|----------|----------|------------|----------|------------|--------------|----------------|\n')
            for i in range(24):
                f.write(f'| {i+1} | {cost_grid_buy_list.iloc[i,0]:.0f} | {revenue_grid_sell_list.iloc[i,0]:.0f} | {revenue_h2_sell_list.iloc[i,0]:.0f} | {revenue_carbon_list.iloc[i,0]:.0f} | {cost_p_bes_list.iloc[i,0]:.2f} | {cost_elec_list.iloc[i,0]:.0f} | {cost_fc_list.iloc[i,0]:.0f} | {money_cost_list.iloc[i,0]:.2f} |\n')
            
            f.write('\n## 三、日汇总\n\n')
            total_buy = cost_grid_buy_list.iloc[:,0].sum()
            total_sell = revenue_grid_sell_list.iloc[:,0].sum()
            total_h2 = revenue_h2_sell_list.iloc[:,0].sum()
            total_carbon = revenue_carbon_list.iloc[:,0].sum()
            total_profit = money_cost_list.iloc[:,0].sum()
            total_profit_real = total_profit * 100
            total_shortage = p_shortage_df.iloc[:,0].sum()
            total_curtail = p_curtail_df.iloc[:,0].sum()
            f.write(f'- 购电成本: {total_buy:.0f} 元\n')
            f.write(f'- 售电收益: {total_sell:.0f} 元\n')
            f.write(f'- 售氢收益: {total_h2:.0f} 元\n')
            f.write(f'- 碳交易收益: {total_carbon:.0f} 元\n')
            f.write(f'- 总缺电量: {total_shortage:.2f} MWh\n')
            f.write(f'- 总弃电量: {total_curtail:.2f} MWh\n')
            f.write(f'- **日净收益: {total_profit_real:.0f} 元** (显示值: {total_profit:.2f} × 100)\n')

