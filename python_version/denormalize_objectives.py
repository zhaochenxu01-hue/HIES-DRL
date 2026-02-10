"""
将归一化目标值转换为实际值
对应 MATLAB: denormalize_objectives.m
"""
from optimize_parameters import C_ref_offgrid, LPSP_ref, LREG_ref


def denormalize_objectives(norm_cost, norm_lpsp, norm_lreg):
    """
    将归一化目标值转换为实际值
    输入:
        norm_cost  - 归一化成本
        norm_lpsp  - 归一化LPSP
        norm_lreg  - 归一化LREG
    输出:
        actual_cost - 实际成本 (元/年)
        actual_lpsp - 实际LPSP (小数)
        actual_lreg - 实际LREG (小数)
    """
    actual_cost = norm_cost * C_ref_offgrid
    actual_lpsp = norm_lpsp * LPSP_ref
    actual_lreg = norm_lreg * LREG_ref
    return actual_cost, actual_lpsp, actual_lreg
