"""
将归一化目标值转换为实际值（离网模式）
功能：计算代表性方案的实际成本、LPSP和LREG
对应 MATLAB: convert_normalized_to_actual.m
"""
import numpy as np
import optimize_parameters as params


def convert_normalized_to_actual():
    """运行代表性方案的实际值转换分析"""

    C_ref = params.C_ref_offgrid

    def crf(r, n):
        return r * (r + 1) ** n / ((r + 1) ** n - 1)

    print('\n' + '=' * 77)
    print('                    代表性方案实际值转换分析')
    print('=' * 77)
    print(f'\n离网模式归一化参考基准:')
    print(f'  成本参考值: {C_ref:.2e} 元/年 ({C_ref / 1e4:.0f} 万元/年)')
    print(f'  LPSP参考值: {params.LPSP_ref * 100:.2f}%')
    print(f'  LREG参考值: {params.LREG_ref * 100:.2f}%')

    # 定义四个代表性方案
    solutions = [
        {'name': '最低成本解',
         'config': [2567, 9092, 8573, 2453, 781, 717],
         'objectives_norm': [0.423, 0.321, 0.0371]},
        {'name': '最低LPSP解',
         'config': [5697, 9648, 9157, 133, 1737, 1425],
         'objectives_norm': [0.581, 0.054, 0.1736]},
        {'name': '最低LREG解',
         'config': [2580, 6542, 7104, 1861, 1073, 782],
         'objectives_norm': [0.508, 0.431, 0.0000]},
        {'name': '膝点解',
         'config': [4172, 7849, 6446, 2406, 837, 859],
         'objectives_norm': [0.487, 0.503, 0.0000]},
    ]

    for i, sol in enumerate(solutions):
        print(f'\n{"=" * 77}')
        print(f'方案{i + 1}: {sol["name"]}')
        print(f'{"=" * 77}')

        ee_bat, p_pv, p_wt, p_elec, h2_storage, p_fc = sol['config']

        print(f'\n【配置参数】')
        print(f'  储能容量:      {ee_bat:7.0f} kWh')
        print(f'  光伏容量:      {p_pv:7.0f} kW')
        print(f'  风电容量:      {p_wt:7.0f} kW')
        print(f'  电解槽容量:    {p_elec:7.0f} kW')
        print(f'  储氢容量:      {h2_storage:7.0f} kg')
        print(f'  燃料电池容量:  {p_fc:7.0f} kW')

        # 投资成本
        print(f'\n【投资成本分解（年化）】')
        inv_bat = crf(params.rp, params.rbat) * params.cbat * ee_bat
        inv_pv = crf(params.rp, params.rPV) * params.cPV * p_pv
        inv_wt = crf(params.rp, params.rWT) * params.cWT * p_wt
        inv_elec = crf(params.rp, params.r_elec) * params.c_elec * p_elec
        inv_h2s = crf(params.rp, params.r_h2storage) * params.c_h2storage * h2_storage
        inv_fc = crf(params.rp, params.r_fc) * params.c_fc * p_fc

        for name, val in [('储能', inv_bat), ('光伏', inv_pv), ('风电', inv_wt),
                          ('电解槽', inv_elec), ('储氢罐', inv_h2s), ('燃料电池', inv_fc)]:
            print(f'  {name}投资:{"":>{8-len(name)*2}}  {val:10.0f} 元/年 ({val / 1e4:.2f} 万元/年)')

        total_inv = inv_bat + inv_pv + inv_wt + inv_elec + inv_h2s + inv_fc
        print(f'  {"─" * 40}')
        print(f'  投资成本合计:  {total_inv:10.0f} 元/年 ({total_inv / 1e4:.2f} 万元/年)')

        # 成本分析
        cost_norm = sol['objectives_norm'][0]
        total_cost = cost_norm * C_ref
        op_cost = total_cost - total_inv

        print(f'\n【成本分析】')
        print(f'  归一化成本:    {cost_norm:.4f}')
        print(f'  实际总成本:    {total_cost:10.0f} 元/年 ({total_cost / 1e4:.2f} 万元/年)')
        print(f'  其中:')
        print(f'    - 投资成本:  {total_inv:10.0f} 元/年 ({total_inv / 1e4:.2f} 万元/年, '
              f'占{total_inv / total_cost * 100:.1f}%)')
        print(f'    - 运行成本:  {op_cost:10.0f} 元/年 ({op_cost / 1e4:.2f} 万元/年, '
              f'占{op_cost / total_cost * 100:.1f}%)')

        # LPSP分析
        lpsp_norm = sol['objectives_norm'][1]
        lpsp_actual = lpsp_norm * params.LPSP_ref

        print(f'\n【LPSP分析（缺电率）】')
        print(f'  归一化LPSP:    {lpsp_norm:.4f}')
        print(f'  实际LPSP:      {lpsp_actual * 100:.4f}% ({lpsp_actual:.6f})')
        if lpsp_actual < 0.005:
            print(f'  评价：优秀，供电可靠性极高')
        elif lpsp_actual < 0.01:
            print(f'  评价：良好，满足离网系统可靠性要求')
        else:
            print(f'  评价：需改进，缺电率较高')

        # LREG分析
        lreg_norm = sol['objectives_norm'][2]
        lreg_actual = lreg_norm * params.LREG_ref

        print(f'\n【LREG分析（弃电率）】')
        print(f'  归一化LREG:    {lreg_norm:.4f}')
        print(f'  实际LREG:      {lreg_actual * 100:.4f}% ({lreg_actual:.6f})')
        if lreg_actual < 0.05:
            print(f'  评价：优秀，可再生能源利用率高')
        elif lreg_actual < 0.15:
            print(f'  评价：良好，弃电率在合理范围')
        else:
            print(f'  评价：需改进，弃电率较高')

        # 估算发电量
        pv_gen = p_pv * 0.15 * 8760
        wt_gen = p_wt * 0.25 * 8760
        total_re = pv_gen + wt_gen
        curtailment = total_re * lreg_actual
        utilized = total_re - curtailment

        print(f'  估算年发电量:')
        print(f'    - 光伏发电:  {pv_gen:10.0f} kWh/年 (容量因子15%)')
        print(f'    - 风电发电:  {wt_gen:10.0f} kWh/年 (容量因子25%)')
        print(f'    - 可再生总:  {total_re:10.0f} kWh/年')
        print(f'    - 弃电量:    {curtailment:10.0f} kWh/年')
        print(f'    - 利用电量:  {utilized:10.0f} kWh/年')

        # 经济性指标
        print(f'\n【经济性指标】')
        if utilized > 0:
            lcoe = total_cost / utilized
            print(f'  度电成本(LCOE): {lcoe:.4f} 元/kWh')
        total_cap = p_pv + p_wt
        unit_inv = total_inv / total_cap
        print(f'  单位容量投资:   {unit_inv:.0f} 元/kW/年')

        # 系统性能
        print(f'\n【系统性能指标】')
        print(f'  可再生能源利用率: {(1 - lreg_actual) * 100:.2f}%')
        print(f'  供电可靠性:       {(1 - lpsp_actual) * 100:.4f}%')

        # 存储结果
        sol['actual_cost'] = total_cost
        sol['investment_cost'] = total_inv
        sol['operation_cost'] = op_cost
        sol['lpsp_actual'] = lpsp_actual
        sol['lreg_actual'] = lreg_actual
        sol['curtailment'] = curtailment
        sol['renewable_utilization'] = (1 - lreg_actual) * 100
        sol['supply_reliability'] = (1 - lpsp_actual) * 100
        sol['utilized_energy'] = utilized

    # 对比总结
    print(f'\n\n{"=" * 77}')
    print(f'                         四个方案对比总结')
    print(f'{"=" * 77}\n')
    header = f'{"指标":<16} | {"最低成本":>12} | {"最低LPSP":>12} | {"最低LREG":>12} | {"膝点解":>12}'
    print(header)
    print('-' * len(header))

    print(f'{"总成本":<16} | {solutions[0]["actual_cost"]/1e4:>10.2f}万 | '
          f'{solutions[1]["actual_cost"]/1e4:>10.2f}万 | '
          f'{solutions[2]["actual_cost"]/1e4:>10.2f}万 | '
          f'{solutions[3]["actual_cost"]/1e4:>10.2f}万')

    print(f'{"LPSP(缺电率)":<16} | {solutions[0]["lpsp_actual"]*100:>11.4f}% | '
          f'{solutions[1]["lpsp_actual"]*100:>11.4f}% | '
          f'{solutions[2]["lpsp_actual"]*100:>11.4f}% | '
          f'{solutions[3]["lpsp_actual"]*100:>11.4f}%')

    print(f'{"LREG(弃电率)":<16} | {solutions[0]["lreg_actual"]*100:>11.4f}% | '
          f'{solutions[1]["lreg_actual"]*100:>11.4f}% | '
          f'{solutions[2]["lreg_actual"]*100:>11.4f}% | '
          f'{solutions[3]["lreg_actual"]*100:>11.4f}%')

    print(f'{"可再生利用率":<16} | {solutions[0]["renewable_utilization"]:>11.2f}% | '
          f'{solutions[1]["renewable_utilization"]:>11.2f}% | '
          f'{solutions[2]["renewable_utilization"]:>11.2f}% | '
          f'{solutions[3]["renewable_utilization"]:>11.2f}%')

    print('=' * 77)

    # 推荐
    print(f'\n【推荐方案分析】\n')
    print(f'1. 最低成本解：适用经济性优先场景')
    print(f'2. 最低LPSP解：适用可靠性优先场景')
    print(f'3. 最低LREG解：适用能源利用效率优先场景')
    print(f'4. 膝点解（推荐）：三者平衡，性价比最高')
    print('=' * 77)


if __name__ == '__main__':
    convert_normalized_to_actual()
