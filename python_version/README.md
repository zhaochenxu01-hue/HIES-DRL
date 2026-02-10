# 离网微电网双层优化 - Python版本

## 项目说明

将MATLAB版本的离网微电网双层优化代码完整转换为Python实现。

**优化目标**：同时优化年化成本、LPSP(缺电率)、LREG(弃电率)三个目标  
**建模方案**：下层小惩罚 + 上层纯三目标（NSGA-II多目标优化）

## 技术栈

| 功能 | MATLAB原工具 | Python替代 |
|------|-------------|-----------|
| 优化建模 | YALMIP | **Pyomo** |
| MILP求解器 | CPLEX | **CPLEX** / HiGHS / Gurobi |
| 多目标GA | gamultiobj | **pymoo (NSGA-II)** |
| 数值计算 | MATLAB内置 | **NumPy / SciPy** |
| 数据读写 | xlsread | **pandas / openpyxl** |
| 可视化 | MATLAB plot | **Matplotlib** |

## 安装

```bash
cd python_version
pip install -r requirements.txt
```

### 求解器配置

默认使用 **CPLEX**。如果没有CPLEX许可证，可切换为免费的 **HiGHS**：

1. 打开 `main_nsga2.py`，修改第23行：
   ```python
   SOLVER_NAME = 'appsi_highs'  # 改为 HiGHS
   ```

2. HiGHS 已包含在 `requirements.txt` 中（`highspy`包）。

如使用 CPLEX，需单独安装：
```bash
pip install cplex
```

## 运行

```bash
python main_nsga2.py
```

确保 `四个典型日数据.xlsx` 位于上级目录中。

## 文件对应关系

| Python文件 | MATLAB原文件 | 功能 |
|-----------|-------------|------|
| `optimize_parameters.py` | `optimize_paremeters.m` | 全局参数定义 |
| `solve_lower_level.py` | `solve_lower_level_no_penalty.m` | 下层MILP调度求解 (Pyomo) |
| `evaluate_fitness.py` | `evaluate_fitness_NSGA2.m` | 适应度评估函数 |
| `generate_lhs_population.py` | `generate_lhs_population.m` | 拉丁超立方采样 |
| `evaluate_pareto_performance.py` | `evaluate_pareto_performance.m` | Pareto前沿评价 + TOPSIS |
| `denormalize_objectives.py` | `denormalize_objectives.m` | 归一化→实际值转换 |
| `convert_normalized_to_actual.py` | `convert_normalized_to_actual.m` | 代表性方案分析 |
| `main_nsga2.py` | `main_NSGA2.m` | 主程序入口 |
| `plot_dispatch_soc_area.py` | `plot_dispatch_soc_area.m` | 功率调度+SOC可视化 |
| `plot_hydrogen_system_analysis.py` | `plot_hydrogen_system_analysis.m` | 制氢系统分析图 |
| `plot_island_dispatch.py` | `plot_island_dispatch.m` | 孤岛模式调度图 |
| `plot_power_balance_symmetrical.py` | `plot_power_balance_symmetrical.m` | 功率平衡对称图 |
| `export_price_emission_data.py` | `export_price_emission_data.m` | 数据导出 |

> `plotMeanFitness.m` 未单独转换，pymoo内置了进化过程可视化功能。

## 注意事项

1. **求解器选项**：`solve_lower_level.py` 中针对不同求解器（CPLEX/HiGHS/Gurobi）设置了对应的参数名。切换求解器后参数会自动适配。
2. **数据路径**：Excel数据文件需位于 `python_version/` 的上级目录。
3. **数值精度**：由于Python和MATLAB的浮点运算差异，结果可能存在微小数值差异（通常 < 0.1%）。
4. **并行计算**：当前版本为串行评估。如需并行，可在 `main_nsga2.py` 中使用 pymoo 的 `StarmapParallelization`。
