import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from agents import AgentBase
import gymnasium as gym
import os
import copy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from tqdm import tqdm
import time

class ReplayBuffer:
    """Inicializa o buffer com tamanho máximo e dimensões dos dados
            
        Args:
            max_size (int): Tamanho máximo do buffer
            state_dim (int): Dimensão do espaço de estados
            action_dim (int): Dimensão do espaço de ações
        """
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Criando arrays para armazenar as experiências
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        """" Adiciona uma nova experiência no buffer

        Args:
            state (np.ndarray): Estado atual
            action (np.ndarray): Ação tomada
            reward (float): Recompensa recebida
            next_state (np.ndarray): Próximo estado
            done (bool): Indica se a experiência terminou
        """

        # Armazena a experiência na posição atual do ponteiro
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size # Circular buffer
        self.size = min(self.size + 1, self.max_size) # Atualiza a quantidade de experiências no buffer

    def sample(self, batch_size):
        """"Amostra um batch de experiências aleatoriamente

        Args:
            batch_size (int): Tamanho do batch a ser amostrado

        Returns:
            tuple: Tuple contendo os batches de estado, ação, recompensa, próximo estado e se a experiência terminou
        """

        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

        

class Actor(nn.Module):
    """Rede neural que mapeia estados para ações"""
    def __init__(self, state_dim, action_dim, max_action):
        """Inicializa a rede do Actor
        
        Args:
            state_dim (int): Dimensão do espaço de estados
            action_dim (int): Dimensão do espaço de ações
            max_action (float): Valor máximo da ação (para escalar a saída)
        """
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim) # saída

        self.max_action = max_action


    def forward(self, state):
        """"
        Mapeia estado para ação

        Args:
            state (torch.Tensor): Estado do agente

        Returns:
            torch.Tensor: Ação gerada pela rede [batch_size, action_dim]
        """

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Camada de saída com tanh para limitar entre [-1, 1]
        # Multiplicamos por max_action para escalar para o intervalo correto
        return self.max_action * torch.tanh(self.l3(a))



class Critic(nn.Module):
    """Rede neural que avalia pares estado-ação"""
    def __init__(self, state_dim, action_dim):
        """Inicializa a rede do Critic
        
        Args:
            state_dim (int): Dimensão do espaço de estados
            action_dim (int): Dimensão do espaço de ações
        """
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256 + action_dim, 256)
        self.l3 = nn.Linear(256, 1) # (Valor que queremos predizer)

    def forward(self, state, action):
        """
        Estima o valor Q para um par estado-ação
        
        Args:
            state (torch.Tensor): Estado [batch_size, state_dim]
            action (torch.Tensor): Ação [batch_size, action_dim]
            
        Returns:
            torch.Tensor: Valor Q estimado [batch_size, 1]
        """

        # Processa o estado
        q = F.relu(self.l1(state))
        # Concatena a ação
        q = torch.cat([q, action], dim=1)  # concatena no eixo 1 (features)
        # Processa estado+ação
        q = F.relu(self.l2(q))
        # Saída final
        return self.l3(q) # Retorna o valor Q estimado
        
class OUNoise:
    """Ornstein-Uhlenbeck process.
    
    Gera ruído temporalmente correlacionado, útil para exploração
    em ambientes com física (como robôs).
    """
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        """
        Args:
            action_dim (int): Dimensão do espaço de ações
            mu (float): Valor médio para convergência
            theta (float): Velocidade de convergência para a média
            sigma (float): Escala do ruído
        """

        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        """Reinicia o processo OU"""
        self.state = self.mu * np.ones(self.action_dim)

    def sample(self):
        """Gera uma amostra de ruído"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class GaussianNoise:
    """Ruído Gaussiano simples.
    
    Mais simples que OU, mas frequentemente igualmente efetivo.
    """
    def __init__(self, action_dim, std=0.1):
        """
        Args:
            action_dim (int): Dimensão do espaço de ações
            std (float): Desvio padrão do ruído
        """
        self.action_dim = action_dim
        self.std = std
        
    def sample(self):
        """Gera uma amostra de ruído"""
        return np.random.normal(0, self.std, size=self.action_dim)
        
        


class DDPGAgent(AgentBase):
    def train(self, 
              episodes=1000, 
              model_path=None, 
              num_envs=1, 
              batch_size=256, 
              learning_rate=1e-4):
        
        # Configurações iniciais
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_timesteps = 25_000
        self.tau = 0.005
        self.gamma = 0.99
        
        # Cria diretório para salvar os modelos se não existir
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        print("\n=== Configuração do Treinamento ===")
        print(f"Total de episódios: {episodes}")
        print(f"Número de ambientes: {num_envs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print("================================\n")

        # Criação dos ambientes
        env = SubprocVecEnv([self._make_env for _ in range(num_envs)])
        eval_env = gym.make(self.env_name)
        new_logger = configure(self.log_dir, ["tensorboard"])  # Removido stdout para reduzir logs

        # Inicialização das redes
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        # Inicializa as redes principais
        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        if model_path:
            print(f"Carregando modelo de: {model_path}")
            self.actor.load_state_dict(torch.load(f"{model_path}_actor.pth"))
            self.critic.load_state_dict(torch.load(f"{model_path}_critic.pth"))
            self.actor_target = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)

        # Inicializa otimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Inicializa replay buffer e noise
        self.replay_buffer = ReplayBuffer(1_000_000, state_dim, action_dim)
        self.noise = GaussianNoise(action_dim)
        
        print("\n=== Iniciando Treinamento ===")
        start_time = time.time()
        
        obs = env.reset()
        episode_num = 0
        best_reward = float('-inf')
        last_eval_step = 0
        eval_freq = 5000  # Mesma frequência do SAC
        
        progress_bar = tqdm(total=episodes, desc="Training")
        t = 0
        while t < episodes:
            # Seleciona ação
            if t < self.start_timesteps:
                action = np.array([env.action_space.sample() for _ in range(num_envs)])
            else:
                action = np.array([
                    self.select_action(obs[i]) for i in range(num_envs)
                ])

            # Executa ação
            next_obs, rewards, dones, infos = env.step(action)

            # Armazena transições no buffer
            for i in range(num_envs):
                self.replay_buffer.add(
                    obs[i], 
                    action[i], 
                    rewards[i], 
                    next_obs[i], 
                    float(dones[i])
                )

            obs = next_obs
            t += 1
            progress_bar.update(1)

            # Treina o agente
            if t >= self.start_timesteps and self.replay_buffer.size > batch_size:
                metrics = self.train_step(batch_size)
                
                # Log silencioso para TensorBoard
                if t % 100 == 0:
                    new_logger.record("train/actor_loss", metrics['actor_loss'])
                    new_logger.record("train/critic_loss", metrics['critic_loss'])
                    new_logger.record("train/timesteps", t)
                    new_logger.record("train/episodes", episode_num)
                    new_logger.record("train/replay_buffer_size", self.replay_buffer.size)
                    
                    # Métricas do ruído de exploração
                    noise_scale = np.mean(np.abs(self.noise.sample()))
                    new_logger.record("train/exploration_noise", noise_scale)
                    
                    # Métricas das ações
                    actions = self.actor(torch.FloatTensor(self.replay_buffer.state[:100]).to(self.device))
                    new_logger.record("train/action_mean", actions.mean().item())
                    new_logger.record("train/action_std", actions.std().item())
                    
                    new_logger.dump(t)

                # Avaliação periódica
                if t - last_eval_step >= eval_freq:
                    last_eval_step = t
                    
                    # Use menos episódios para avaliação (5 em vez de 10)
                    n_eval_episodes = 5
                    eval_rewards = []
                    eval_lengths = []
                    success_rate = 0
                    
                    # Avaliação mais rápida
                    with torch.no_grad():  # Evita cálculo de gradientes durante avaliação
                        for _ in range(n_eval_episodes):
                            done = False
                            eval_obs = eval_env.reset()[0]
                            eval_reward = 0
                            eval_length = 0
                            
                            while not done:
                                eval_action = self.select_action(eval_obs, evaluate=True)
                                eval_obs, r, done, _, info = eval_env.step(eval_action)
                                eval_reward += r
                                eval_length += 1
                                
                                # Se houver informação de sucesso no ambiente
                                if 'is_success' in info:
                                    success_rate += info['is_success']
                            
                            eval_rewards.append(eval_reward)
                            eval_lengths.append(eval_length)
                    
                    mean_reward = np.mean(eval_rewards)
                    std_reward = np.std(eval_rewards)
                    mean_length = np.mean(eval_lengths)
                    success_rate = success_rate / n_eval_episodes
                    
                    # Métricas de avaliação
                    new_logger.record("eval/mean_reward", mean_reward)
                    new_logger.record("eval/reward_std", std_reward)
                    new_logger.record("eval/mean_episode_length", mean_length)
                    new_logger.record("eval/success_rate", success_rate)
                    
                    # Métricas do modelo - computadas apenas uma vez para economizar tempo
                    if t > self.start_timesteps:
                        sample_states, _, _, _, _ = self.replay_buffer.sample(min(100, self.replay_buffer.size))
                        with torch.no_grad():
                            sample_actions = self.actor(sample_states)
                        
                        new_logger.record("eval/action_mean", sample_actions.mean().item())
                        new_logger.record("eval/action_std", sample_actions.std().item())
                    
                    new_logger.dump(t)
                    
                    # Salva apenas se melhorou
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        print(f"\nNova melhor recompensa média: {best_reward:.2f} (± {std_reward:.2f})")
                        if success_rate > 0:
                            print(f"Taxa de sucesso: {success_rate*100:.1f}%")
                        torch.save(self.actor.state_dict(), f"{self.model_path}_best_actor.pth")
                        torch.save(self.critic.state_dict(), f"{self.model_path}_best_critic.pth")

            if dones[0]:
                episode_num += 1
                obs = env.reset()

        # Salva o modelo final
        torch.save(self.actor.state_dict(), f"{self.model_path}_final_actor.pth")
        torch.save(self.critic.state_dict(), f"{self.model_path}_final_critic.pth")
        
        elapsed_time = time.time() - start_time
        print("\n=== Treinamento Concluído ===")
        print(f"Tempo total: {elapsed_time/60:.1f} minutos")
        print(f"Steps/segundo: {episodes/elapsed_time:.1f}")
        print(f"Melhor recompensa: {best_reward:.2f}")
        print("============================\n")
        
        env.close()
        eval_env.close()


    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.actor(state).cpu().numpy()
        
        if not evaluate:
            noise = self.noise.sample()  # Corrigido: estava usando 'noise' sem definir
            action = action + noise
        
        return np.clip(action, -self.max_action, self.max_action)

        

    def train_step(self, batch_size):
        """Realiza um passo de treinamento usando um batch do replay buffer"""

        # Sample do replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # Atualiza o Critic
        with torch.no_grad():
            # Seleciona próxima ação usando a rede target
            next_action = self.actor_target(next_state)

            # Computa o target Q-value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Computa o Q-value atual
        current_Q = self.critic(state, action)

        # Calcula o erro de Q-learning
        loss = F.mse_loss(current_Q, target_Q)

        # Otimiza o Critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # Atualiza o Actor  
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Atualiza as redes target
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {
            'critic_loss': loss.item(),
            'actor_loss': actor_loss.item()
        }

    
    def _soft_update(self, target, source):
        """Atualização suave dos parâmetros da rede target"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def eval(self):
        """Avalia o agente em modo determinístico (sem exploração)"""
        # Verifica se os arquivos do modelo existem
        actor_path = f"{self.model_path}_actor.pth"
        if not os.path.exists(actor_path):
            print(f"\nModelo não encontrado em {actor_path}")
            print("Por favor, treine o agente primeiro.")
            return
        
        env = gym.make(self.env_name, render_mode="human")
        
        # Carrega o modelo
        self.actor = Actor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=float(env.action_space.high[0])
        ).to(self.device)
        
        # Carrega os pesos salvos
        self.actor.load_state_dict(torch.load(actor_path))
        
        # Coloca em modo de avaliação
        self.actor.eval()
        
        obs, _ = env.reset()
        rewards = []
        
        for _ in range(1000):
            # Seleciona ação sem ruído (determinístico)
            action = self.select_action(obs, evaluate=True)
            
            # Executa a ação
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            
            # Renderiza o ambiente
            env.render()
            
            # Reset se o episódio terminou
            if done:
                obs, _ = env.reset()
        
        env.close()


                    
                







