"""
PPO算法模块 — 用于上层容量规划
复用自 PPO+IES-1h.py 的 Actor/Critic/PPO 架构
改动:
  1. 状态维度/动作维度参数化（适配容量规划MDP）
  2. 保留 Tanh-Squash + GAE(λ) + Entropy Bonus + Advantage Normalization
  3. 增加学习率调度（余弦退火）
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as params


class Actor(nn.Module):
    """策略网络：输出容量调整动作的均值与标准差"""

    def __init__(self, state_dim, action_dim, hidden_dim=None):
        super(Actor, self).__init__()
        hidden_dim = hidden_dim or params.HIDDEN_DIM

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu_head(x)
        std = torch.exp(torch.clamp(self.log_std_head, -2.0, 0.5))
        return mu, std


class Critic(nn.Module):
    """价值网络：估计状态价值 V(s)"""

    def __init__(self, state_dim, hidden_dim=None):
        super(Critic, self).__init__()
        hidden_dim = hidden_dim or params.HIDDEN_DIM

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value


class PPOAgent:
    """
    PPO算法主类
    改进: GAE(λ) + Entropy Bonus + Advantage Normalization + Tanh-Squash
    """

    def __init__(self, state_dim, action_dim, action_bound_high, action_bound_low,
                 device=None):
        self.device = device or torch.device("cpu")

        self.action_bound_high = torch.FloatTensor(action_bound_high).to(self.device)
        self.action_bound_low = torch.FloatTensor(action_bound_low).to(self.device)
        self.action_dim = action_dim
        self.state_dim = state_dim

        # 创建网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=params.LR_ACTOR
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=params.LR_CRITIC
        )

        # 学习率调度器（余弦退火）
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=params.TRAIN_EPISODES, eta_min=1e-6
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=params.TRAIN_EPISODES, eta_min=1e-6
        )

        # 经验缓冲区
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []

        # GAE参数
        self.gae_lambda = params.GAE_LAMBDA
        self.entropy_coef = params.ENTROPY_COEF

    def store_transition(self, state, action, reward, done, value, log_prob):
        """存储经验到缓冲区"""
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.value_buffer.append(value)
        self.log_prob_buffer.append(log_prob)

    def get_action(self, state, greedy=False):
        """
        根据状态选择动作 (Tanh-Squash)
        返回: action (numpy), value (float), log_prob (float)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, std = self.actor(state_tensor)
            value = self.critic(state_tensor).item()

            if greedy:
                u = mu
                action_tanh = torch.tanh(u)
                action = (action_tanh * (self.action_bound_high - self.action_bound_low) / 2
                          + (self.action_bound_high + self.action_bound_low) / 2)
                log_prob = 0.0
            else:
                dist = Normal(mu, std)
                u = dist.rsample()

                action_tanh = torch.tanh(u)
                action = (action_tanh * (self.action_bound_high - self.action_bound_low) / 2
                          + (self.action_bound_high + self.action_bound_low) / 2)

                log_prob_raw = dist.log_prob(u).sum(dim=1)
                log_prob_correction = torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=1)
                log_prob = (log_prob_raw - log_prob_correction).item()

        return action.cpu().detach().numpy().flatten(), value, log_prob

    def compute_gae(self, last_value, done):
        """计算GAE(λ) advantage"""
        rewards = np.array(self.reward_buffer)
        values = np.array(self.value_buffer + [last_value])
        dones = np.array(self.done_buffer)

        advantages = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + params.GAMMA * values[t + 1] * mask - values[t]
            gae = delta + params.GAMMA * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + np.array(self.value_buffer)
        return advantages, returns

    def update(self, last_state, done):
        """更新策略网络和价值网络"""
        if len(self.state_buffer) == 0:
            return {}

        # 计算最后状态的价值
        with torch.no_grad():
            last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            last_value = self.critic(last_state_tensor).item() if not done else 0

        # 计算GAE
        advantages, returns = self.compute_gae(last_value, done)

        # 转换为tensor
        states = torch.FloatTensor(np.array(self.state_buffer)).to(self.device)
        actions = torch.FloatTensor(np.array(self.action_buffer)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_prob_buffer)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

        # Advantage normalization
        if advantages_tensor.std() > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO Actor更新
        actor_losses = []
        for _ in range(params.ACTOR_UPDATE_STEPS):
            mu, std = self.actor(states)
            dist = Normal(mu, std)

            # 从action反推u
            action_normalized = ((actions - (self.action_bound_high + self.action_bound_low) / 2)
                                 / ((self.action_bound_high - self.action_bound_low) / 2 + 1e-8))
            action_normalized = torch.clamp(action_normalized, -0.999, 0.999)
            u = torch.atanh(action_normalized)

            log_prob_raw = dist.log_prob(u).sum(dim=1)
            log_prob_correction = torch.log(1 - action_normalized.pow(2) + 1e-6).sum(dim=1)
            new_log_probs = log_prob_raw - log_prob_correction

            entropy = dist.entropy().sum(dim=1).mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - params.EPSILON_CLIP, 1 + params.EPSILON_CLIP) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), params.GRAD_CLIP)
            self.actor_optimizer.step()
            actor_losses.append(actor_loss.item())

        # Critic更新
        critic_losses = []
        for _ in range(params.CRITIC_UPDATE_STEPS):
            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns_tensor)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), params.GRAD_CLIP)
            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())

        # 学习率调度
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # 清空缓冲区
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.value_buffer.clear()
        self.log_prob_buffer.clear()

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'lr_actor': self.actor_scheduler.get_last_lr()[0],
        }

    def save(self, path=None):
        """保存模型"""
        if path is None:
            os.makedirs(params.MODEL_DIR, exist_ok=True)
            path = params.MODEL_DIR
        torch.save(self.actor.cpu().state_dict(), os.path.join(path, 'ppo_actor.pth'))
        torch.save(self.critic.cpu().state_dict(), os.path.join(path, 'ppo_critic.pth'))
        self.actor.to(self.device)
        self.critic.to(self.device)

    def load(self, path=None):
        """加载模型"""
        if path is None:
            path = params.MODEL_DIR
        self.actor.load_state_dict(
            torch.load(os.path.join(path, 'ppo_actor.pth'), map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(path, 'ppo_critic.pth'), map_location=self.device)
        )
