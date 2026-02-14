"""
多场景生成器
功能:
  1. 从Excel加载4典型日基准数据
  2. 通过Data Augmentation生成多场景训练数据（负荷/风光波动）
  3. 计算场景统计特征（用于构建PPO状态）
"""
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as params


class ScenarioGenerator:
    """多场景生成器：加载基准数据 + 随机扰动生成训练/测试场景"""

    def __init__(self, noise_level=0.10):
        """
        参数:
            noise_level - 场景扰动幅度，默认±10%
        """
        self.noise_level = noise_level
        self.base_data = None
        self._load_base_data()

    def _load_base_data(self):
        """从Excel加载4典型日基准数据"""
        excel_file = params.EXCEL_FILE
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f'数据文件不存在: {excel_file}')

        sheet_name = '0%'
        df_load = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='B:E',
                                skiprows=2, nrows=24, header=None)
        df_wt = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='H:K',
                              skiprows=2, nrows=24, header=None)
        df_pv = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='N:Q',
                              skiprows=2, nrows=24, header=None)

        self.base_data = {
            'p_load': df_load.values.astype(float),        # 24×4
            'p_pv_percent': df_pv.values.astype(float),    # 24×4
            'p_wt_percent': df_wt.values.astype(float),    # 24×4
        }

    def sample_scenario(self):
        """
        生成一个带随机扰动的场景
        返回: external_data dict (格式与下层求解器兼容)
        """
        p_load = self.base_data['p_load'].copy()
        p_pv = self.base_data['p_pv_percent'].copy()
        p_wt = self.base_data['p_wt_percent'].copy()

        # 对每个典型日独立施加乘性高斯扰动
        for d in range(4):
            load_noise = 1.0 + np.random.uniform(-self.noise_level, self.noise_level, 24)
            pv_noise = 1.0 + np.random.uniform(-self.noise_level, self.noise_level, 24)
            wt_noise = 1.0 + np.random.uniform(-self.noise_level, self.noise_level, 24)

            p_load[:, d] = np.maximum(p_load[:, d] * load_noise, 0)
            p_pv[:, d] = np.clip(p_pv[:, d] * pv_noise, 0, 1)
            p_wt[:, d] = np.clip(p_wt[:, d] * wt_noise, 0, 1)

        return {
            'p_load': p_load,
            'p_pv_percent': p_pv,
            'p_wt_percent': p_wt,
        }

    def get_base_scenario(self):
        """返回无扰动的基准场景"""
        return {
            'p_load': self.base_data['p_load'].copy(),
            'p_pv_percent': self.base_data['p_pv_percent'].copy(),
            'p_wt_percent': self.base_data['p_wt_percent'].copy(),
        }

    @staticmethod
    def compute_scenario_features(external_data, investment_vars=None):
        """
        计算场景统计特征（用于构建PPO状态向量）
        输入:
            external_data   - 场景数据 dict
            investment_vars - 当前容量配置 [6]，可选
        输出:
            features - 特征向量 (numpy array)
        """
        p_load = external_data['p_load']       # 24×4
        p_pv = external_data['p_pv_percent']   # 24×4
        p_wt = external_data['p_wt_percent']   # 24×4

        N_days = np.array(params.N_days)  # [91, 92, 91, 91]
        total_days = N_days.sum()  # 365

        # --- 负荷特征 (6维) ---
        # 年加权均值和统计量
        load_weighted = np.sum(p_load * N_days[np.newaxis, :], axis=1) / total_days  # 24维
        p_load_mean = np.mean(load_weighted)
        p_load_max = np.max(load_weighted)
        p_load_std = np.std(load_weighted)

        # 净负荷统计（假设容量=1时的百分比净负荷）
        # 此处用百分比数据的均值来反映风光资源禀赋
        pv_weighted = np.sum(p_pv * N_days[np.newaxis, :], axis=1) / total_days
        wt_weighted = np.sum(p_wt * N_days[np.newaxis, :], axis=1) / total_days

        # 负荷与可再生百分比的相对关系
        net_load_profile = load_weighted  # 此处净负荷需要实际容量，先用负荷代替
        p_net_mean = np.mean(net_load_profile)
        p_net_max = np.max(net_load_profile)
        p_net_min = np.min(net_load_profile)

        # --- 可再生能源特征 (4维) ---
        pv_cf = np.mean(pv_weighted)  # 光伏年均利用率
        wt_cf = np.mean(wt_weighted)  # 风电年均利用率
        pv_peak_ratio = np.max(pv_weighted) / (np.mean(pv_weighted) + 1e-6)
        wt_peak_ratio = np.max(wt_weighted) / (np.mean(wt_weighted) + 1e-6)

        features = np.array([
            p_load_mean / 10000.0,   # 归一化到合理范围
            p_load_max / 10000.0,
            p_load_std / 5000.0,
            p_net_mean / 10000.0,
            p_net_max / 10000.0,
            p_net_min / 10000.0,
            pv_cf,
            wt_cf,
            pv_peak_ratio / 5.0,
            wt_peak_ratio / 5.0,
        ], dtype=np.float32)

        return features
