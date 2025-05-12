# agents/cleanrl_sac_agent.py

import json
import random
import time
from distutils.util import strtobool
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from agents import AgentBase

# Constantes do SAC
LOG_STD_MAX = 2
LOG_STD_MIN = -5

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, 256))
        self.fc_mean = layer_init(nn.Linear(256, np.prod(env.action_space.shape)))
        self.fc_logstd = layer_init(nn.Linear(256, np.prod(env.action_space.shape)))
        
        self.register_buffer(
            "action_scale",
            torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0)
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        # Adiciona dimensão do batch se necessário
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        mean, log_std = self(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Calcula log_prob
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)  # Soma ao longo da dimensão da ação
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, 256))  # Adicionada mais uma camada
        self.fc4 = layer_init(nn.Linear(256, 1))

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    def __init__(self, buffer_size, obs_space, action_space, device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        self.obs = np.zeros((buffer_size,) + obs_space.shape, dtype=np.float32)
        self.next_obs = np.zeros((buffer_size,) + obs_space.shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size,) + action_space.shape, dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        idx = np.random.randint(0, upper_bound, size=batch_size)

        return (
            torch.FloatTensor(self.obs[idx]).to(self.device),
            torch.FloatTensor(self.next_obs[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )

class CleanRLSACAgent(AgentBase):
    def train(self, 
            episodes=500000,
            model_path=None,
            num_envs=16,
            batch_size=1024,
            learning_rate=0.001):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=== Configuração do Treinamento ===")
        print(f"Total de episódios: {episodes}")
        print(f"Número de ambientes: {num_envs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {self.device}")
        print("================================\n")

        buffer_size = 1000000
        gamma = 0.99
        tau = 0.005
        alpha = 0.2
        autotune = True

        # Métricas de progresso
        episode_rewards = [0 for _ in range(num_envs)]
        episode_steps = [0 for _ in range(num_envs)]
        episodes_completed = 0
        total_steps = 0
        best_reward = float('-inf')
        rewards_history = []

        # Criação dos ambientes vetorizados
        def make_env():
            def _init():
                env = gym.make(self.env_name)
                return env
            return _init
        
        envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
        
        # Inicialização das redes neurais
        single_env = gym.make(self.env_name)
        actor = Actor(single_env).to(self.device)
        qf1 = SoftQNetwork(single_env).to(self.device)
        qf2 = SoftQNetwork(single_env).to(self.device)
        qf1_target = SoftQNetwork(single_env).to(self.device)
        qf2_target = SoftQNetwork(single_env).to(self.device)
        single_env.close()
        
        # Carregar modelo se existir
        if model_path:
            print(f"Carregando modelo de: {model_path}")
            actor.load_state_dict(torch.load(f"{model_path}_actor.pth"))
            qf1.load_state_dict(torch.load(f"{model_path}_qf1.pth"))
            qf2.load_state_dict(torch.load(f"{model_path}_qf2.pth"))
            qf1_target.load_state_dict(qf1.state_dict())
            qf2_target.load_state_dict(qf2.state_dict())
        else:
            print("Criando novo modelo")

        # Otimizadores
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=learning_rate)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)

        if autotune:
            target_entropy = -float(np.prod(single_env.action_space.shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=learning_rate)

        rb = ReplayBuffer(buffer_size, single_env.observation_space, single_env.action_space, self.device)
        writer = SummaryWriter(self.log_dir)

        print("\n=== Iniciando Treinamento ===")
        start_time = time.time()
        
        obs, _ = envs.reset()
        
        # Usando tqdm para progress bar
        progress_bar = tqdm(range(episodes), desc="Training")
        
        for step in progress_bar:
            total_steps += num_envs

            # Seleção das ações para todos os ambientes
            if step < 10000:
                actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
            else:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                    actions, _, _ = actor.get_action(obs_tensor)
                    actions = actions.cpu().numpy()

            # Execução das ações em todos os ambientes
            next_obs, rewards, dones, truncateds, infos = envs.step(actions)
            
            # Atualiza recompensas e passos para cada ambiente
            for i in range(num_envs):
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                
                rb.add(obs[i], next_obs[i], actions[i], rewards[i], dones[i])
                
                if dones[i] or truncateds[i]:
                    rewards_history.append(episode_rewards[i])
                    avg_reward = sum(rewards_history[-100:]) / min(len(rewards_history), 100)
                    
                    # Atualiza a descrição da barra de progresso
                    progress_bar.set_postfix({
                        'eps': episodes_completed,
                        'reward': f'{episode_rewards[i]:.2f}',
                        'avg100': f'{avg_reward:.2f}'
                    })
                    
                    writer.add_scalar("charts/episode_reward", episode_rewards[i], episodes_completed)
                    writer.add_scalar("charts/episode_length", episode_steps[i], episodes_completed)
                    writer.add_scalar("charts/average_reward", avg_reward, episodes_completed)
                    
                    # Salva o melhor modelo
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        print(f"\nNovo melhor reward médio: {best_reward:.2f}!")
                        torch.save(actor.state_dict(), f"{self.log_dir}/best_model_actor.pth")
                        torch.save(qf1.state_dict(), f"{self.log_dir}/best_model_qf1.pth")
                        torch.save(qf2.state_dict(), f"{self.log_dir}/best_model_qf2.pth")
                    
                    episodes_completed += 1
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                    
                    next_obs_i, _ = envs.reset()
                    next_obs[i] = next_obs_i[0]

            obs = next_obs

            # Treinamento após warm-up
            if step >= 10000:
                for _ in range(num_envs):
                    b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(batch_size)

                    with torch.no_grad():
                        next_actions, next_log_pi, _ = actor.get_action(b_next_obs)
                        q1_next = qf1_target(b_next_obs, next_actions)
                        q2_next = qf2_target(b_next_obs, next_actions)
                        q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                        q_target = b_rewards + gamma * (1 - b_dones) * q_next

                    # Update Q-networks
                    q1_pred = qf1(b_obs, b_actions)
                    q2_pred = qf2(b_obs, b_actions)
                    q1_loss = F.mse_loss(q1_pred, q_target)
                    q2_loss = F.mse_loss(q2_pred, q_target)
                    q_loss = q1_loss + q2_loss

                    q_optimizer.zero_grad()
                    q_loss.backward()
                    q_optimizer.step()

                    # Update actor
                    actions, log_pi, _ = actor.get_action(b_obs)
                    q1_pi = qf1(b_obs, actions)
                    q2_pi = qf2(b_obs, actions)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha * log_pi - min_q_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if autotune:
                        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                    # Update target networks
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Logging a cada 1000 steps
                if step % 1000 == 0:
                    writer.add_scalar("losses/q1_loss", q1_loss.item(), step)
                    writer.add_scalar("losses/q2_loss", q2_loss.item(), step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), step)
                    writer.add_scalar("losses/alpha", alpha, step)
                    if autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), step)

        elapsed_time = time.time() - start_time
        print("\n=== Treinamento Concluído ===")
        print(f"Episódios completados: {episodes_completed}")
        print(f"Total de steps: {total_steps}")
        print(f"Melhor reward médio: {best_reward:.2f}")
        print(f"Tempo total: {elapsed_time/60:.1f} minutos")
        print(f"Steps/segundo: {total_steps/elapsed_time:.1f}")
        print("============================")

        # Salvando modelo final
        torch.save(actor.state_dict(), f"{self.model_path}_actor.pth")
        torch.save(qf1.state_dict(), f"{self.model_path}_qf1.pth")
        torch.save(qf2.state_dict(), f"{self.model_path}_qf2.pth")
        
        envs.close()
        writer.close()

    def eval(self):
        env = gym.make(self.env_name, render_mode="human")
        actor = Actor(env).to(self.device)
        actor.load_state_dict(torch.load(f"{self.model_path}_actor.pth"))
        actor.eval()
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                action = actor.get_action(torch.FloatTensor(obs).to(self.device))[0].cpu().numpy()
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
            if truncated:
                break
                
        env.close()