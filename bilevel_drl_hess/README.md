# 基于深度强化学习PPO的电氢综合能源系统规划运行双层优化

## 概述

本项目实现了一种基于PPO(Proximal Policy Optimization)深度强化学习与CPLEX混合整数线性规划(MILP)的双层优化框架，用于电池-氢气混合储能系统(HESS)的容量规划与运行优化协同决策。

- **上层（PPO）**：容量规划决策 — 确定储能、光伏、风电、电解槽、储氢、燃料电池的最优容量配置
- **下层（CPLEX MILP）**：运行优化调度 — 在给定容量下求解4季典型日×24h的最优调度方案

## 项目结构

```
bilevel_drl_hess/
├── config/
│   └── parameters.py          # 统一参数配置（投资/运维/技术/PPO超参数）
├── data/
│   └── 四个典型日数据.xlsx     # 4季典型日负荷/风光数据
├── lower_level/
│   └── milp_solver.py         # 下层MILP调度求解器（Pyomo + CPLEX）
├── upper_level/
│   ├── ppo_agent.py           # PPO算法（Actor/Critic/GAE/Tanh-Squash）
│   ├── capacity_env.py        # 容量规划Gym环境（MDP建模）
│   └── scenario_generator.py  # 多场景生成器（Data Augmentation）
├── evaluation/
│   ├── reliability.py         # 可靠性评估（LPSP/LREG/成本计算）
│   └── plot_results.py        # 绘图模块（训练曲线/调度图/对比表）
├── main_train.py              # PPO训练主程序
├── main_test.py               # 测试与对比实验（PPO/NSGA-II/GA/PSO）
├── requirements.txt           # 依赖包
└── README.md
```

## 核心技术

### 上层MDP建模

| 要素 | 设计 |
|------|------|
| **状态** (24维) | 场景统计特征(10) + 当前容量配置(6) + 历史最优信息(3) + 进度(1) + 改进信息(4) |
| **动作** (6维) | 容量调整增量 Δx ∈ [-1,1]^6，映射到实际调整量 |
| **奖励** | -(年化总成本/归一化) - LPSP约束惩罚 - LREG约束惩罚 |
| **转移** | x_new = clip(x + Δx × step_size, lb, ub) |

### 下层MILP模型

- 4季典型日×24h离网调度
- 完整约束：储能充放电互斥、电解槽启停/爬坡/死区、燃料电池启停/爬坡、跨季节储氢衔接+年闭合
- VoLL经济化处理：切负荷成本30元/kWh，弃电机会成本0.5元/kWh（替代原惩罚系数）

### PPO算法特性

- Tanh-Squash动作映射 + Jacobian修正
- GAE(λ=0.95) 优势函数估计
- Entropy Bonus正则化
- 余弦退火学习率调度
- 梯度裁剪(0.5)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：需要安装CPLEX求解器。可选替代：`appsi_highs`（免费）或 `gurobi`。

### 2. 训练

```bash
python main_train.py
```

训练产出：
- `models/ppo_actor.pth` / `ppo_critic.pth` — PPO模型权重
- `output/training_curves.png` — 训练收敛曲线
- `output/training_results.json` — 训练过程数据

### 3. 测试与对比实验

```bash
# 完整对比（PPO + NSGA-II + GA + PSO）
python main_test.py

# 仅测试PPO
python main_test.py --ppo-only

# 自定义基线参数
python main_test.py --nsga2-pop 50 --nsga2-gen 30 --ga-gen 100
```

测试产出：
- `output/comparison_results.json` — 对比结果数据
- `output/comparison_table.png` — 对比表格
- `output/capacity_comparison.png` — 容量配置对比
- `output/dispatch_4seasons.png` — 四季典型日功率调度
- `output/h2_seasonal_linkage.png` — 跨季节储氢SOC衔接

## 对比实验设计

| Case | 上层方法 | 下层方法 | 说明 |
|------|---------|---------|------|
| Case 1 | PPO | CPLEX MILP | 本文方法 |
| Case 2 | NSGA-II | CPLEX MILP | 多目标进化算法基线 |
| Case 3 | GA | CPLEX MILP | 遗传算法基线 |
| Case 4 | PSO | CPLEX MILP | 粒子群优化基线 |

## 关键参数说明

在 `config/parameters.py` 中可调整：

- **PPO超参数**：`TRAIN_EPISODES`, `N_STEPS_PER_EPISODE`, `LR_ACTOR`, `EPSILON_CLIP` 等
- **容量搜索空间**：`CAP_LOWER`, `CAP_UPPER`, `STEP_RATIO`
- **可靠性约束**：`MAX_LPSP_ALLOWED`(5%), `MAX_LREG_ALLOWED`(15%)
- **求解器**：`SOLVER_NAME`（支持 cplex/gurobi/appsi_highs）

## 参考文献

1. Qian et al., "Co-Optimization of Capacity and Operation for Battery-Hydrogen Hybrid Energy Storage System," *Energies*, 2025. (TD3+Gurobi MIP)
2. Li et al., "Bi-Level Adaptive Storage Expansion Strategy for Microgrids Using Deep Reinforcement Learning," *IEEE Trans. Smart Grid*, 2024. (Rainbow-QR+LP)
