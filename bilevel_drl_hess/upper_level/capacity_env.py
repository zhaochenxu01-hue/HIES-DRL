"""
容量规划Gym环境
功能:
  将电氢综合能源系统容量规划建模为马尔可夫决策过程(MDP)
  - State: 场景特征 + 当前配置(归一化) + 历史信息
  - Action: 6维连续容量调整增量 Δx ∈ [-1, 1]^6
  - Reward: -(年化总成本) - 可靠性约束惩罚
  - Transition: x_new = clip(x_current + Δx * step_size, lb, ub)

每个Episode对应一次容量搜索过程(N_STEPS_PER_EPISODE步)
每步调用下层CPLEX MILP求解器获得运行成本和可靠性指标
"""
import numpy as np
import gym
from gym import spaces

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as params
from lower_level.milp_solver import solve_lower_level
from upper_level.scenario_generator import ScenarioGenerator


class CapacityPlanningEnv(gym.Env):
    """
    上层容量规划环境

    状态空间 (24维):
        [0:10]  场景统计特征 (负荷/风光/净负荷统计量)
        [10:16] 当前容量配置 (归一化到[0,1])
        [16:19] 历史最优信息 (最优成本/LPSP/LREG, 归一化)
        [19]    当前步数进度 (t / N_steps)
        [20:24] 上一步改进信息 (成本变化/LPSP变化/LREG变化/改进率)

    动作空间 (6维):
        Δx ∈ [-1, 1]^6, 映射到容量调整: x += Δx * step_size

    Reward:
        约束化单目标: min 年化总成本, s.t. LPSP ≤ 5%, LREG ≤ 15%
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, noise_level=0.10, solver_name=None):
        super(CapacityPlanningEnv, self).__init__()

        self.solver_name = solver_name or params.SOLVER_NAME
        self.scenario_gen = ScenarioGenerator(noise_level=noise_level)

        # 容量边界
        self.cap_lower = np.array(params.CAP_LOWER, dtype=np.float32)
        self.cap_upper = np.array(params.CAP_UPPER, dtype=np.float32)
        self.cap_range = self.cap_upper - self.cap_lower
        self.n_cap = params.N_CAP_VARS  # 6

        # 步长: 每步最大调整量
        self.step_size = self.cap_range * params.STEP_RATIO

        # 动作空间: [-1, 1]^6
        self.action_space = spaces.Box(
            low=-np.ones(self.n_cap, dtype=np.float32),
            high=np.ones(self.n_cap, dtype=np.float32),
            dtype=np.float32
        )

        # 状态空间: 24维
        self.state_dim = 24
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32
        )

        # 最大步数
        self.max_steps = params.N_STEPS_PER_EPISODE

        # 内部状态
        self.current_cap = None
        self.current_scenario = None
        self.scenario_features = None
        self.t = 0
        self.best_cost = np.inf
        self.best_lpsp = 1.0
        self.best_lreg = 1.0
        self.best_cap = None
        self.prev_total_cost = None
        self.prev_lpsp = None
        self.prev_lreg = None
        self.improvement_count = 0
        self.no_improve_steps = 0

        # 容量评估缓存（同一episode内复用）
        self._eval_cache = {}
        self._cache_grid = self.cap_range * params.CACHE_GRID_RATIO

        # 投资成本CRF函数
        self._crf_cache = {}

    def _crf(self, r, n):
        """等额年金回收系数"""
        key = (r, n)
        if key not in self._crf_cache:
            self._crf_cache[key] = r * (r + 1) ** n / ((r + 1) ** n - 1)
        return self._crf_cache[key]

    def _compute_investment_cost(self, cap):
        """计算年化投资成本"""
        inv_traditional = (self._crf(params.rp, params.rbat) * params.cbat * cap[0]
                           + self._crf(params.rp, params.rPV) * params.cPV * cap[1]
                           + self._crf(params.rp, params.rWT) * params.cWT * cap[2])
        inv_hydrogen = (self._crf(params.rp, params.r_elec) * params.c_elec * cap[3]
                        + self._crf(params.rp, params.r_h2storage) * params.c_h2storage * cap[4]
                        + self._crf(params.rp, params.r_fc) * params.c_fc * cap[5])
        return inv_traditional + inv_hydrogen

    def _compute_reliability(self, detailed_results):
        """从下层求解结果计算LPSP和LREG"""
        total_load = detailed_results.get('total_load', 1e-6)
        total_shed = detailed_results.get('total_shed', 0)
        total_curtail = detailed_results.get('total_curtail', 0)
        total_renewable = detailed_results.get('total_renewable', 1e-6)

        lpsp = total_shed / max(total_load, 1e-6)
        lreg = total_curtail / max(total_renewable, 1e-6)
        return lpsp, lreg

    def _normalize_cap(self, cap):
        """将容量归一化到[0,1]"""
        return (cap - self.cap_lower) / (self.cap_range + 1e-6)

    def _build_state(self):
        """构建24维状态向量"""
        cap_norm = self._normalize_cap(self.current_cap)

        # 历史最优信息 (归一化)
        best_cost_norm = min(self.best_cost / params.C_ref, 5.0)
        best_lpsp_norm = min(self.best_lpsp / params.LPSP_ref, 5.0)
        best_lreg_norm = min(self.best_lreg / params.LREG_ref, 5.0)

        # 步数进度
        progress = self.t / self.max_steps

        # 上一步改进信息
        if self.prev_total_cost is not None and self.best_cost < np.inf:
            cost_change = (self.prev_total_cost - self.best_cost) / (params.C_ref + 1e-6)
            lpsp_change = (self.prev_lpsp - self.best_lpsp) / (params.LPSP_ref + 1e-6)
            lreg_change = (self.prev_lreg - self.best_lreg) / (params.LREG_ref + 1e-6)
            imp_rate = self.improvement_count / max(self.t, 1)
        else:
            cost_change = 0.0
            lpsp_change = 0.0
            lreg_change = 0.0
            imp_rate = 0.0

        state = np.concatenate([
            self.scenario_features,           # 10维
            cap_norm,                          # 6维
            np.array([best_cost_norm, best_lpsp_norm, best_lreg_norm]),  # 3维
            np.array([progress]),              # 1维
            np.array([cost_change, lpsp_change, lreg_change, imp_rate]),  # 4维
        ]).astype(np.float32)

        state = np.clip(state, -10.0, 10.0)
        return state

    def _cap_to_key(self, cap):
        """将容量按网格量化后转tuple作为缓存key"""
        quantized = np.round(cap / (self._cache_grid + 1e-6)) * self._cache_grid
        return tuple(quantized.astype(np.float32))

    def _evaluate_capacity(self, cap):
        """
        评估一组容量配置（带缓存）
        返回: (total_cost, op_cost, lpsp, lreg, feasible)
        """
        cache_key = self._cap_to_key(cap)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        investment_cost = self._compute_investment_cost(cap)

        op_cost, detailed_results = solve_lower_level(
            cap, self.current_scenario, solver_name=self.solver_name
        )

        if op_cost > 1e11 or not detailed_results:
            result = (np.inf, op_cost, 1.0, 1.0, False)
        else:
            lpsp, lreg = self._compute_reliability(detailed_results)
            total_cost = investment_cost + op_cost
            result = (total_cost, op_cost, lpsp, lreg, True)

        self._eval_cache[cache_key] = result
        return result

    def reset(self):
        """
        环境重置:
        1. 采样新场景
        2. 随机初始化容量配置
        3. 评估初始配置
        """
        # 采样场景
        self.current_scenario = self.scenario_gen.sample_scenario()
        self.scenario_features = ScenarioGenerator.compute_scenario_features(self.current_scenario)

        # 随机初始化容量（LHS采样式均匀分布）
        self.current_cap = (self.cap_lower
                            + np.random.rand(self.n_cap) * self.cap_range).astype(np.float32)

        self.t = 0
        self.improvement_count = 0
        self.no_improve_steps = 0
        self._eval_cache = {}  # 每个episode清空缓存（场景不同）

        # 评估初始配置
        total_cost, _, lpsp, lreg, feasible = self._evaluate_capacity(self.current_cap)

        if feasible:
            self.best_cost = total_cost
            self.best_lpsp = lpsp
            self.best_lreg = lreg
        else:
            self.best_cost = np.inf
            self.best_lpsp = 1.0
            self.best_lreg = 1.0

        self.best_cap = self.current_cap.copy()
        self.prev_total_cost = self.best_cost
        self.prev_lpsp = self.best_lpsp
        self.prev_lreg = self.best_lreg

        return self._build_state()

    def step(self, action):
        """
        执行一步容量调整
        参数:
            action: [-1, 1]^6 容量调整比例
        返回:
            obs, reward, done, info
        """
        self.t += 1
        action = np.clip(action, -1.0, 1.0)

        # 记录上一步信息
        self.prev_total_cost = self.best_cost if self.best_cost < np.inf else params.C_ref * 5
        self.prev_lpsp = self.best_lpsp
        self.prev_lreg = self.best_lreg

        # 容量更新: x_new = x_current + Δx * step_size
        cap_new = self.current_cap + action * self.step_size
        cap_new = np.clip(cap_new, self.cap_lower, self.cap_upper).astype(np.float32)
        self.current_cap = cap_new

        # 评估新配置
        total_cost, op_cost, lpsp, lreg, feasible = self._evaluate_capacity(cap_new)

        # ==================== Reward计算 ====================
        if not feasible:
            reward = params.INFEASIBLE_PENALTY
            self.no_improve_steps += 1
        else:
            # 年化总成本（归一化）
            cost_norm = total_cost / params.REWARD_SCALE

            # LPSP约束惩罚（barrier函数形式）
            lpsp_penalty = 0.0
            if lpsp > params.MAX_LPSP_ALLOWED:
                lpsp_penalty = params.LAMBDA_LPSP * (lpsp - params.MAX_LPSP_ALLOWED) ** 2

            # LREG约束惩罚
            lreg_penalty = 0.0
            if lreg > params.MAX_LREG_ALLOWED:
                lreg_penalty = params.LAMBDA_LREG * (lreg - params.MAX_LREG_ALLOWED) ** 2

            reward = -(cost_norm + lpsp_penalty + lreg_penalty)

            # 更新最优记录（仅当满足可靠性约束时）
            if (total_cost < self.best_cost and
                    lpsp <= params.MAX_LPSP_ALLOWED and
                    lreg <= params.MAX_LREG_ALLOWED):
                self.best_cost = total_cost
                self.best_lpsp = lpsp
                self.best_lreg = lreg
                self.best_cap = cap_new.copy()
                self.improvement_count += 1
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1

        # 终止条件
        done = (self.t >= self.max_steps or
                self.no_improve_steps >= params.NO_IMPROVE_PATIENCE)

        # 构建info
        info = {
            'total_cost': total_cost if feasible else np.inf,
            'op_cost': op_cost if feasible else np.inf,
            'investment_cost': self._compute_investment_cost(cap_new),
            'lpsp': lpsp,
            'lreg': lreg,
            'feasible': feasible,
            'current_cap': cap_new.copy(),
            'best_cost': self.best_cost,
            'best_cap': self.best_cap.copy() if self.best_cap is not None else None,
            'best_lpsp': self.best_lpsp,
            'best_lreg': self.best_lreg,
            'step': self.t,
            'improvement_count': self.improvement_count,
        }

        obs = self._build_state()
        return obs, reward, done, info

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def render(self, mode='human'):
        if mode == 'human':
            cap = self.current_cap
            print(f'Step {self.t}/{self.max_steps} | '
                  f'储能={cap[0]:.0f} 光伏={cap[1]:.0f} 风电={cap[2]:.0f} '
                  f'电解槽={cap[3]:.0f} 储氢={cap[4]:.0f} FC={cap[5]:.0f} | '
                  f'Best Cost={self.best_cost:.0f} LPSP={self.best_lpsp:.4f} LREG={self.best_lreg:.4f}')


class CapacityPlanningEnvDeterministic(CapacityPlanningEnv):
    """
    确定性版本：使用固定基准场景（用于测试和对比）
    """

    def reset(self):
        self.current_scenario = self.scenario_gen.get_base_scenario()
        self.scenario_features = ScenarioGenerator.compute_scenario_features(self.current_scenario)

        self.current_cap = (self.cap_lower
                            + np.random.rand(self.n_cap) * self.cap_range).astype(np.float32)

        self.t = 0
        self.improvement_count = 0
        self.no_improve_steps = 0
        self._eval_cache = {}  # 每个episode清空缓存

        total_cost, _, lpsp, lreg, feasible = self._evaluate_capacity(self.current_cap)

        if feasible:
            self.best_cost = total_cost
            self.best_lpsp = lpsp
            self.best_lreg = lreg
        else:
            self.best_cost = np.inf
            self.best_lpsp = 1.0
            self.best_lreg = 1.0

        self.best_cap = self.current_cap.copy()
        self.prev_total_cost = self.best_cost
        self.prev_lpsp = self.best_lpsp
        self.prev_lreg = self.best_lreg

        return self._build_state()
