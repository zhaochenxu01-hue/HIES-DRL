import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置绘图风格，适配中文
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 适配中文
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-paper')  # 使用科研论文风格


def plot_models_for_proposal():
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('电-氢综合能源系统关键设备精细化建模', fontsize=20, y=0.95)

    # ==========================================
    # 子图1: 储能电池非线性充放电效率曲线
    # ==========================================
    ax1 = fig.add_subplot(2, 2, 1)

    p_ratio = np.linspace(0, 1, 100)
    # 模拟非线性效率：低功率辅助损耗大，高功率内阻热损耗大，中间效率最高
    eff_ch = 0.96 - 0.05 * (p_ratio - 0.4) ** 2 - 0.02 / (p_ratio + 0.1)
    eff_dis = 0.96 - 0.05 * (p_ratio - 0.4) ** 2 - 0.02 / (p_ratio + 0.1)

    ax1.plot(p_ratio, eff_ch, 'r-', linewidth=2.5, label='充电效率 $\eta_{ch}$')
    ax1.plot(p_ratio, eff_dis, 'b--', linewidth=2.5, label='放电效率 $\eta_{dis}$')

    # 对比现有线性模型
    ax1.axhline(0.95, color='gray', linestyle=':', label='传统线性模型 (0.95)')

    ax1.set_xlabel('充/放电功率比 (P/P_rated)', fontsize=12)
    ax1.set_ylabel('效率', fontsize=12)
    ax1.set_title('(a) 储能电池非线性效率模型', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ==========================================
    # 子图2: 电池循环寿命模型 (Wöhler Curve / DOD)
    # ==========================================
    ax2 = fig.add_subplot(2, 2, 2)

    dod = np.linspace(10, 100, 50)  # 放电深度 %
    # 典型的锂电池寿命公式：Cycle = A * DOD^(-k)
    cycles = 3000 * (dod / 100) ** (-1.5)

    ax2.plot(dod, cycles, 'g-o', linewidth=2, markersize=4)
    ax2.fill_between(dod, cycles, color='green', alpha=0.1)

    ax2.set_xlabel('放电深度 DOD (%)', fontsize=12)
    ax2.set_ylabel('最大循环次数 (Cycles)', fontsize=12)
    ax2.set_title('(b) 电池寿命衰减模型 (基于雨流计数法基础)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    # 添加注释：说明这是雨流计数的物理基础
    ax2.text(60, 4000, '浅充浅放寿命更长\n深充深放寿命急剧下降',
             bbox=dict(facecolor='white', alpha=0.8))

    # ==========================================
    # 子图3: 电解槽/燃料电池 变效率曲线 (极化特性)
    # ==========================================
    ax3 = fig.add_subplot(2, 2, 3)

    x = np.linspace(0, 1, 100)

    # 模拟真实的极化曲线导致的效率变化
    # 电解槽：低负荷效率高(电压低)，但考虑辅助功耗后，极低负荷效率骤降
    # 这里模拟包含BoP损耗的系统效率
    eff_elec = 0.70 * (1 - 0.1 * (1 - x) ** 2) * (1 - np.exp(-10 * x))  # 最后一项模拟启动死区

    # 燃料电池：低负荷活化极化损耗大，高负荷浓差极化大
    eff_fc = 0.60 * (1 - 0.15 * (x - 0.3) ** 2) * (1 - np.exp(-15 * x))

    ax3.plot(x, eff_elec, 'teal', linewidth=2.5, label='电解槽效率 (PEM)')
    ax3.plot(x, eff_fc, 'orange', linewidth=2.5, label='燃料电池效率 (PEMFC)')

    # 标注死区和最佳工作区
    ax3.axvspan(0, 0.1, color='gray', alpha=0.2, label='启动/死区约束')
    ax3.axvspan(0.3, 0.8, color='yellow', alpha=0.1, label='高效工作区')

    ax3.set_xlabel('负载率 (P/P_rated)', fontsize=12)
    ax3.set_ylabel('系统效率 (LHV)', fontsize=12)
    ax3.set_title('(c) 氢能设备变效率与运行约束', fontsize=14)
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ==========================================
    # 子图4: 储氢系统能耗修正 (引入压缩机)
    # ==========================================
    ax4 = fig.add_subplot(2, 2, 4)

    m_dot = np.linspace(0, 50, 100)  # 产氢速率 kg/h

    # 传统模型：无额外能耗
    p_comp_old = np.zeros_like(m_dot)

    # 升级模型：考虑压缩功耗 (假设从30bar压到350bar, 绝热压缩)
    # 粗略估算：每kg氢气耗电 1.5 - 3 kWh
    spec_energy = 2.0  # kWh/kg
    p_comp_new = m_dot * spec_energy

    ax4.plot(m_dot, p_comp_old, 'k--', label='传统模型 (忽略压缩)')
    ax4.plot(m_dot, p_comp_new, 'r-', linewidth=2.5, label='精细化模型 (含压缩机能耗)')

    ax4.fill_between(m_dot, p_comp_old, p_comp_new, color='red', alpha=0.1, label='修正能耗偏差')

    ax4.set_xlabel('产氢速率 (kg/h)', fontsize=12)
    ax4.set_ylabel('压缩机附加功耗 (kW)', fontsize=12)
    ax4.set_title('(d) 储氢环节能耗修正模型', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(pad=3.0)
    plt.savefig('modeling_upgrade_proposal.png', dpi=300)
    print("图片已保存为: modeling_upgrade_proposal.png")


if __name__ == "__main__":
    plot_models_for_proposal()