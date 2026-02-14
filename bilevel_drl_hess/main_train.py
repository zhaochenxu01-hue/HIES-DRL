"""
PPO-CPLEX双层优化训练主程序
上层: PPO容量规划Agent
下层: CPLEX MILP运行优化求解器
"""
import os
import sys
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import parameters as params
from upper_level.ppo_agent import PPOAgent
from upper_level.capacity_env import CapacityPlanningEnv
from evaluation.reliability import print_solution_summary, evaluate_solution
from evaluation.plot_results import plot_training_curves


def train():
    """PPO训练主循环"""
    print('=' * 80)
    print('  基于深度强化学习PPO的电氢综合能源系统规划运行双层优化')
    print('  上层: PPO容量规划  |  下层: CPLEX MILP运行优化')
    print('=' * 80)

    # ========== 1. 设置随机种子 ==========
    seed = params.RANDOM_SEED
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ========== 2. 创建环境 ==========
    print('\n[1/4] 初始化环境...')
    env = CapacityPlanningEnv(noise_level=0.10, solver_name=params.SOLVER_NAME)
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound_high = env.action_space.high
    action_bound_low = env.action_space.low

    print(f'  状态维度: {state_dim}')
    print(f'  动作维度: {action_dim}')
    print(f'  容量下限: {params.CAP_LOWER}')
    print(f'  容量上限: {params.CAP_UPPER}')
    print(f'  每Episode步数: {params.N_STEPS_PER_EPISODE}')
    print(f'  训练Episodes: {params.TRAIN_EPISODES}')

    # ========== 3. 创建PPO Agent ==========
    print('\n[2/4] 创建PPO Agent...')
    agent = PPOAgent(state_dim, action_dim, action_bound_high, action_bound_low)
    print(f'  Actor网络: {sum(p.numel() for p in agent.actor.parameters())} 参数')
    print(f'  Critic网络: {sum(p.numel() for p in agent.critic.parameters())} 参数')

    # ========== 4. 训练循环 ==========
    print('\n[3/4] 开始训练...')
    os.makedirs(params.OUTPUT_DIR, exist_ok=True)
    os.makedirs(params.MODEL_DIR, exist_ok=True)

    # 训练记录
    episode_rewards = []
    best_costs_history = []
    best_lpsps_history = []
    best_lregs_history = []
    global_best_cost = np.inf
    global_best_cap = None
    global_best_lpsp = 1.0
    global_best_lreg = 1.0
    total_milp_calls = 0

    t_start = time.time()
    log_file = os.path.join(params.OUTPUT_DIR, 'training_log.txt')

    for episode in range(params.TRAIN_EPISODES):
        state = env.reset()
        episode_reward = 0.0
        ep_milp_calls = 0

        for step in range(params.N_STEPS_PER_EPISODE):
            # PPO选择动作
            action, value, log_prob = agent.get_action(state)

            # 执行动作
            state_next, reward, done, info = env.step(action)
            ep_milp_calls += 1

            # 存储经验
            agent.store_transition(state, action, reward, done, value, log_prob)

            state = state_next
            episode_reward += reward

            if done:
                break

        # Episode结束，PPO更新
        update_info = agent.update(state_next, done)
        total_milp_calls += ep_milp_calls

        # 记录该Episode的最优结果
        ep_best_cost = info.get('best_cost', np.inf)
        ep_best_lpsp = info.get('best_lpsp', 1.0)
        ep_best_lreg = info.get('best_lreg', 1.0)
        ep_best_cap = info.get('best_cap', None)

        # 更新全局最优
        if (ep_best_cost < global_best_cost and
                ep_best_lpsp <= params.MAX_LPSP_ALLOWED and
                ep_best_lreg <= params.MAX_LREG_ALLOWED):
            global_best_cost = ep_best_cost
            global_best_cap = ep_best_cap.copy() if ep_best_cap is not None else None
            global_best_lpsp = ep_best_lpsp
            global_best_lreg = ep_best_lreg

        episode_rewards.append(episode_reward)
        best_costs_history.append(global_best_cost if global_best_cost < np.inf else 0)
        best_lpsps_history.append(global_best_lpsp)
        best_lregs_history.append(global_best_lreg)

        # 打印进度
        elapsed = time.time() - t_start
        actor_loss = update_info.get('actor_loss', 0)
        critic_loss = update_info.get('critic_loss', 0)
        lr = update_info.get('lr_actor', params.LR_ACTOR)

        log_msg = (f'Ep {episode+1:4d}/{params.TRAIN_EPISODES} | '
                   f'Reward={episode_reward:8.2f} | '
                   f'EpBest={ep_best_cost/1e4:8.1f}万 | '
                   f'GlobalBest={global_best_cost/1e4:8.1f}万 | '
                   f'LPSP={ep_best_lpsp*100:5.2f}% | '
                   f'LREG={ep_best_lreg*100:5.2f}% | '
                   f'Imp={info.get("improvement_count",0):2d} | '
                   f'MILP={ep_milp_calls:3d} | '
                   f'ALoss={actor_loss:7.4f} CLoss={critic_loss:7.4f} | '
                   f'LR={lr:.1e} | '
                   f'Time={elapsed:.0f}s')
        print(log_msg)

        # 写入日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

        # 定期保存模型
        if (episode + 1) % 100 == 0:
            agent.save()
            print(f'  [Checkpoint] 模型已保存 (Episode {episode+1})')

        # 定期保存训练曲线
        if (episode + 1) % 200 == 0:
            try:
                plot_training_curves(episode_rewards, best_costs_history,
                                     best_lpsps_history, best_lregs_history, save=True)
                plt_closed = True
            except Exception:
                plt_closed = False
            if plt_closed:
                import matplotlib.pyplot as plt
                plt.close('all')

    # ========== 5. 训练完成 ==========
    total_time = time.time() - t_start
    print('\n' + '=' * 80)
    print('训练完成！')
    print(f'  总耗时: {total_time:.1f}秒 ({total_time/3600:.2f}小时)')
    print(f'  总MILP调用次数: {total_milp_calls}')
    print(f'  全局最优成本: {global_best_cost/1e4:.2f}万元/年')
    print(f'  全局最优LPSP: {global_best_lpsp*100:.4f}%')
    print(f'  全局最优LREG: {global_best_lreg*100:.4f}%')

    if global_best_cap is not None:
        print(f'  最优容量配置:')
        for i, name in enumerate(params.CAP_NAMES):
            print(f'    {name}: {global_best_cap[i]:.0f}')

    # 保存最终模型
    agent.save()
    print(f'\n[4/4] 模型已保存至 {params.MODEL_DIR}')

    # 保存训练曲线
    plot_training_curves(episode_rewards, best_costs_history,
                         best_lpsps_history, best_lregs_history, save=True)
    import matplotlib.pyplot as plt
    plt.close('all')

    # 保存训练结果到JSON
    training_results = {
        'total_time': total_time,
        'total_milp_calls': total_milp_calls,
        'global_best_cost': global_best_cost,
        'global_best_lpsp': global_best_lpsp,
        'global_best_lreg': global_best_lreg,
        'global_best_cap': global_best_cap.tolist() if global_best_cap is not None else None,
        'episode_rewards': episode_rewards,
        'best_costs_history': best_costs_history,
    }
    results_file = os.path.join(params.OUTPUT_DIR, 'training_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(training_results, f, ensure_ascii=False, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else o)
    print(f'训练结果已保存至: {results_file}')

    return global_best_cap, global_best_cost, global_best_lpsp, global_best_lreg


if __name__ == '__main__':
    train()
