"""
导出分时电价与碳排放数据到Excel
对应 MATLAB: export_price_emission_data.m
"""
import numpy as np
import pandas as pd
import os


def export_price_emission_data():
    """导出分时电价与碳排放数据"""

    hours = np.arange(1, 25)

    # 购电价 (元/kWh)
    p_buy = np.array([
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.45,
        0.85, 0.85, 0.85, 0.45, 0.45, 0.45, 0.45, 0.45,
        0.45, 0.45, 0.85, 0.85, 0.85, 0.45, 0.35, 0.35
    ])

    # 售电价 (元/kWh)
    p_sell = p_buy * 0.6

    # 碳排放因子 (kgCO2/kWh)
    e_grid = np.array([
        0.85, 0.85, 0.85, 0.85, 0.80, 0.75, 0.70, 0.65,
        0.60, 0.58, 0.56, 0.55, 0.54, 0.55, 0.56, 0.58,
        0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.84, 0.85
    ])

    # 峰谷标记
    is_peak = np.array([
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0
    ])
    peak_label = ['峰时' if p == 1 else '谷/平时' for p in is_peak]

    # 创建DataFrame
    df = pd.DataFrame({
        '时段': hours,
        '购电价_元每kWh': p_buy,
        '售电价_元每kWh': p_sell,
        '碳排放因子_kgCO2每kWh': e_grid,
        '峰谷标记': peak_label,
    })

    # 导出
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(data_dir, '分时电价与碳排放数据.xlsx')
    df.to_excel(filename, index=False, sheet_name='Sheet1')

    print(f'成功导出数据到: {filename}')
    print('数据预览（前5行）:')
    print(df.head())


if __name__ == '__main__':
    export_price_emission_data()
