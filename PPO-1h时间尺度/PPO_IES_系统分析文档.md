# PPO算法优化电氢综合能源系统 - 系统分析文档

## 一、优化问题建模

### 1.1 目标函数

**最大化综合能源系统日运行净收益：**

$$\max J_{total} = \sum_{t=1}^{24} \left( R_{sell}(t) + R_{H2}(t) + R_{carbon}(t) - C_{buy}(t) - C_{op}(t) - C_{deg}(t) \right)$$

各项收益/成本计算：

| 项目 | 计算公式 | 说明 |
|------|----------|------|
| 售电收益 | $R_{sell} = \lambda_{sell}(t) \cdot \|P_{grid}(t)\| \cdot \Delta t$ | 向电网售电 |
| 售氢收益 | $R_{H2} = \lambda_{H2} \cdot m_{H2,sell}(t)$ | 氢气售价30元/kg |
| 碳交易收益 | $R_{carbon} = P_{CO2} \cdot (C_{H2} \cdot m_{H2,sell} + C_{FC} \cdot P_{FC})$ | 碳减排收益 |
| 购电成本 | $C_{buy} = \lambda_{buy}(t) \cdot P_{grid}(t) \cdot \Delta t$ | 从电网购电 |
| 运维成本 | $C_{op} = 0.015 \cdot P_{elec} + 0.020 \cdot P_{FC}$ | 电解槽15、燃料电池20元/MWh |
| 退化成本 | $C_{deg} = 0.1 \cdot \|P_{bes}\| + 0.05 \cdot P_{elec} + 0.03 \cdot P_{FC}$ | 储能100、电解槽50、燃料电池30元/MWh |
| 启停成本 | $C_{startup} = 25 \cdot N_{elec} + 50 \cdot N_{FC}$ | 电解槽25元/次，燃料电池50元/次 |

---

### 1.2 约束处理策略（核心设计）

**设计理念：软约束为主 + 闭环投影**

本系统采用"动作投影 + 软惩罚"的混合约束处理策略：
- **去除了大部分硬性约束**，使策略探索更加自由
- **通过cmd-exec差值惩罚**引导策略学习可行动作
- **仅保留物理必须的硬约束**（如功率平衡、SOC边界、日终闭合）

#### 1.2.1 约束分类

| 约束类型 | 处理方式 | 说明 |
|----------|----------|------|
| 功率上下限 | 软投影 + violation记录 | 超限部分记入cmd-exec惩罚 |
| 爬坡约束 | 软投影 + violation记录 | 超限部分记入cmd-exec惩罚 |
| 死区约束 | 硬截断 | 负荷率<20%强制归零 |
| 储能SOC | 闭环投影 | 自动调整p_bes保证SOC可行 |
| 储氢SOC | 闭环投影 | 超上限削减产氢，超下限削减用氢 |
| 并网约束 | 闭环投影 + 缺电/弃电记录 | 最终钳住p_grid |
| 日终SOC闭合 | 硬约束（t=23） | 强制回归初始SOC |

#### 1.2.2 已删除的硬约束

| 原约束 | 删除原因 |
|--------|----------|
| 储能分时策略（谷充峰放） | 限制策略探索，改为弱偏好引导 |
| 电解槽-燃料电池互斥 | 物理上可同时运行，改为经济引导 |
| 风光充足时FC禁止运行 | 改为弱偏好惩罚 |
| 需量电费硬约束 | 已删除，由并网约束自然限制 |

---

### 1.3 强化学习奖励函数

**奖励函数设计（归一化到[-10, 10]范围）：**

```python
REWARD_SCALE = 5000.0

# 利润项（真实财务）
reward_profit = profit_t / REWARD_SCALE

# 引导项（弱偏好）
guide_bonus = (bonus_valley_elec + bonus_elec_surplus - cost_peak_elec - penalty_fc_surplus) / REWARD_SCALE

# 惩罚项（仅保留关键惩罚，避免重复）
total_punishment = punishment_shortage + punishment_curtail + punishment_soc_closure + punishment_cmd_exec
reward_punishment = -total_punishment / REWARD_SCALE

# 总奖励
reward = reward_profit + guide_bonus + reward_punishment

# 日终SOC闭合惩罚
if done_next:
    reward -= k_terminal_soc * abs(new_c_soc - 0.5) / REWARD_SCALE
    reward -= k_terminal_h2 * abs(new_h2_soc - 0.5) / REWARD_SCALE
```

**cmd-exec差值惩罚（核心机制）：**

```python
# 归一化各维度，避免MW与kg/h量纲混淆
scale = (action_space.high - action_space.low)
cmd_exec_diff = (|action_cmd - action_exec| / scale).sum()
punishment_cmd_exec = k_violation * cmd_exec_diff  # k_violation = 100
```

---

### 1.4 等式约束（能量平衡）

**电功率平衡：**
$$P_{load}(t) + P_{elec}(t) = P_{grid}(t) + P_{pv}(t) + P_{wt}(t) + P_{bes}(t) + P_{FC}(t)$$

**氢气质量平衡：**
$$SOC_{H2}(t+1) = SOC_{H2}(t) + \frac{m_{H2,prod} - m_{H2,consume} - m_{H2,sell}}{Cap_{H2}}$$

---

### 1.5 不等式约束（设备出力限制）

| 设备 | 约束条件 | 代码参数 |
|------|----------|----------|
| 电网交换 | $-3 \leq P_{grid} \leq 2$ MW | `min_p_grid=-3, max_p_grid=2` |
| 电储能 | $-1.5 \leq P_{bes} \leq 1.5$ MW | `min_p_bes=-1.5, max_p_bes=1.5` |
| 电解槽 | $0 \leq P_{elec} \leq 2$ MW | `min_p_elec=0, max_p_elec=2` |
| 燃料电池 | $0 \leq P_{FC} \leq 2$ MW | `min_p_fc=0, max_p_fc=2` |
| 售氢速率 | $0 \leq m_{sell} \leq 25$ kg/h | `max_h2_sell=25` (5%容量/h) |

---


## 二、PPO求解运行优化核心步骤

### 2.1 算法总体流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PPO求解运行优化核心流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 第一阶段：环境初始化                                                   │  │
│  │  • 创建Environment实例，设置设备参数                                   │  │
│  │  • 创建PPO智能体（Actor + Critic网络）                                │  │
│  │  • 设置随机种子保证可复现                                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 第二阶段：训练循环（10000 episodes × 24 steps）                        │  │
│  │                                                                        │  │
│  │   for episode in range(TRAIN_EPISODES):                               │  │
│  │       state = env.reset()  # 重置环境，添加负荷不确定性                 │  │
│  │                                                                        │  │
│  │       for t in range(24):  # 24小时调度                                │  │
│  │           ┌────────────────────────────────────────────────────────┐  │  │
│  │           │ Step 1: Actor网络输出动作                               │  │  │
│  │           │   mu, std = Actor(state)                               │  │  │
│  │           │   u ~ Normal(mu, std)                                  │  │  │
│  │           │   action = tanh(u) × (high-low)/2 + (high+low)/2       │  │  │
│  │           └────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                         │  │
│  │           ┌────────────────────────────────────────────────────────┐  │  │
│  │           │ Step 2: 动作投影（project_action）                      │  │  │
│  │           │   • 功率上下限约束 → 记录violation                      │  │  │
│  │           │   • 爬坡约束 → 记录violation                           │  │  │
│  │           │   • 死区截断 → 效率计算                                 │  │  │
│  │           │   • 储能SOC闭环投影                                     │  │  │
│  │           │   • 储氢SOC闭环投影                                     │  │  │
│  │           │   • 日终SOC硬闭合（t=23）                               │  │  │
│  │           │   • 并网约束 → 缺电/弃电处理                            │  │  │
│  │           └────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                         │  │
│  │           ┌────────────────────────────────────────────────────────┐  │  │
│  │           │ Step 3: 计算奖励                                        │  │  │
│  │           │   reward = profit + guide_bonus - punishment            │  │  │
│  │           │   • profit: 真实财务收益                                │  │  │
│  │           │   • guide_bonus: 弱偏好引导                             │  │  │
│  │           │   • punishment: cmd-exec差值 + 缺电弃电 + SOC势函数     │  │  │
│  │           └────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                         │  │
│  │           ┌────────────────────────────────────────────────────────┐  │  │
│  │           │ Step 4: 存储经验                                        │  │  │
│  │           │   buffer.store(state, action, reward, value, log_prob) │  │  │
│  │           └────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │       # Episode结束后更新网络                                          │  │
│  │       ┌────────────────────────────────────────────────────────────┐  │  │
│  │       │ Step 5: PPO更新                                             │  │  │
│  │       │   • 计算GAE(λ)优势函数                                      │  │  │
│  │       │   • Advantage归一化                                         │  │  │
│  │       │   • PPO-Clip更新Actor（10次迭代）                           │  │  │
│  │       │   • MSE损失更新Critic（10次迭代）                           │  │  │
│  │       │   • 梯度裁剪（max_norm=0.5）                                │  │  │
│  │       └────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 第三阶段：测试评估                                                     │  │
│  │  • 加载训练好的模型                                                    │  │
│  │  • 使用greedy策略（tanh(mu)）执行调度                                  │  │
│  │  • 输出调度结果、成本分析、可视化图表                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.2 动作投影详细流程（project_action）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        动作投影流程（9步闭环）                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: action_cmd = [p_elec, p_bes, p_fc, h2_sell]                         │
│        c_soc, h2_soc, t                                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: 功率上下限约束                                               │   │
│  │   p_elec = clip(p_elec_cmd, 0, 2)                                   │   │
│  │   p_fc = clip(p_fc_cmd, 0, 2)                                       │   │
│  │   p_bes = clip(p_bes_cmd, -1.5, 1.5)                                │   │
│  │   h2_sell = clip(h2_sell_cmd, 0, 25)                                │   │
│  │   → 记录 violation['elec_power'], violation['fc_power'] 等          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 2: 爬坡约束                                                     │   │
│  │   p_elec_max = last_p_elec + 0.5 × 2MW                              │   │
│  │   p_elec_min = last_p_elec - 0.8 × 2MW                              │   │
│  │   p_elec = clip(p_elec, p_elec_min, p_elec_max)                     │   │
│  │   → 记录 violation['elec_ramp'], violation['fc_ramp']               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 3: 死区约束与效率计算                                           │   │
│  │   if load_ratio < 0.2: p_elec = 0, elec_eff = 0                     │   │
│  │   else: elec_eff = 0.65 × (1 - 0.2 × (1-load_ratio)²)               │   │
│  │   → 变效率建模（极化曲线）                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 4: 储能SOC约束（闭环投影）                                       │   │
│  │   max_charge = (0.9 - c_soc) × 6MWh / 0.95                          │   │
│  │   max_discharge = (c_soc - 0.1) × 6MWh × 0.95                       │   │
│  │   p_bes = clip(p_bes, -max_charge, max_discharge)                   │   │
│  │   → 保证SOC不越界                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 5: 售氢库存约束                                                  │   │
│  │   h2_sell = min(h2_sell, h2_soc × 500kg × 0.9)                      │   │
│  │   → 不能卖空气                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 6: 计算氢气产量和消耗                                            │   │
│  │   h2_prod = p_elec × elec_eff / 33.33 × 1000  (kg/h)                │   │
│  │   h2_consume = p_fc / fc_eff / 33.33 × 1000   (kg/h)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 7: 储氢SOC约束（闭环投影）                                       │   │
│  │   new_h2_soc = h2_soc + (h2_prod - h2_consume - h2_sell) / 500      │   │
│  │   if new_h2_soc > 0.9:                                              │   │
│  │       削减h2_prod → 反推p_elec → 重算效率                            │   │
│  │   elif new_h2_soc < 0.1:                                            │   │
│  │       削减h2_sell → 削减h2_consume → 反推p_fc                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 7.5: 日终SOC硬闭合（t=23时触发）                                 │   │
│  │   储能: 计算p_bes_target使c_soc回归soc_init                          │   │
│  │   储氢: 调整h2_sell使h2_soc回归h2_soc_init                           │   │
│  │   → 记录violation，保证日循环调度闭合                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 8: 功率平衡与并网约束                                            │   │
│  │   p_grid = p_load + p_elec - p_pv - p_wt - p_bes - p_fc             │   │
│  │   if p_grid > 2MW:                                                  │   │
│  │       尝试增加储能放电 → 重算p_grid                                   │   │
│  │       if still > 2MW: p_shortage = p_grid - 2, p_grid = 2           │   │
│  │   elif p_grid < -3MW:                                               │   │
│  │       尝试增加储能充电 → 重算p_grid                                   │   │
│  │       if still < -3MW: p_curtail = -3 - p_grid, p_grid = -3         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 9: 闭环一致性计算                                                │   │
│  │   new_c_soc = c_soc - n_bes × p_bes × Δt / Q_bes                    │   │
│  │   new_h2_soc = h2_soc + (h2_prod - h2_consume - h2_sell) / Cap_h2   │   │
│  │   → 最终SOC由实际执行动作决定，不做clip                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  输出: action_exec, violation, aux_info                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.3 PPO算法核心机制

#### 2.3.1 Tanh-Squash动作映射

```python
# Actor网络输出原始mu（无界）
mu, std = Actor(state)

# 训练时：重参数化采样 + Tanh压缩
u = mu + std * epsilon  # epsilon ~ N(0,1)
action_tanh = tanh(u)   # 压缩到[-1, 1]
action = action_tanh * (high - low) / 2 + (high + low) / 2

# 测试时（greedy）：同样走Tanh+缩放
u = mu
action_tanh = tanh(u)
action = action_tanh * (high - low) / 2 + (high + low) / 2

# Log-prob修正（Jacobian）
log_prob = log_prob_raw - sum(log(1 - tanh²(u) + 1e-6))
```

#### 2.3.2 GAE(λ)优势函数

```python
def compute_gae(rewards, values, dones, gamma=0.98, lambda_=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1-dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

#### 2.3.3 PPO-Clip更新

```python
# 计算新旧策略比率
ratio = exp(new_log_prob - old_log_prob)

# Clip目标函数
surr1 = ratio * advantage
surr2 = clip(ratio, 1-ε, 1+ε) * advantage
actor_loss = -min(surr1, surr2).mean() - entropy_coef * entropy

# Advantage归一化
advantage = (advantage - mean) / (std + 1e-8)
```

---

