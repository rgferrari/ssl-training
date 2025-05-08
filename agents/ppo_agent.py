import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn
from torch.optim import Adam
import time

from agents import AgentBase


class PPOAgent(AgentBase):
    def train(self, 
              episodes=1000, 
              model_path=None, 
              num_envs=1, 
              batch_size=256, 
              learning_rate=1e-4):
        print("\n=== Configuração do Treinamento ===")
        print(f"Total de episódios: {episodes}")
        print(f"Número de ambientes: {num_envs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print("================================\n")

        env = SubprocVecEnv([self._make_env for _ in range(num_envs)])
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])

        if model_path:
            print(f"Carregando modelo de: {model_path}")
            model = PPO.load(
                path=model_path, 
                env=env,
                device='cuda'    
            )
        else:
            print("Criando novo modelo")
            model = PPO(
                'MlpPolicy',
                env,
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=[256, 256, 256],
                        vf=[256, 256, 256]
                    ),
                    activation_fn=nn.ReLU,
                    optimizer_class=Adam
                ),
                n_steps=2048,  # Número de steps por atualização
                batch_size=batch_size, 
                n_epochs=10,    # Número de epochs por atualização
                gamma=0.99,     # Fator de desconto
                gae_lambda=0.95, # GAE lambda parameter
                clip_range=0.2,  # Clip parameter for PPO
                clip_range_vf=None,  # Clip parameter for value function
                normalize_advantage=True,
                ent_coef=0.01,  # Coeficiente de entropia fixo
                vf_coef=0.5,    # Coeficiente da função valor
                max_grad_norm=0.5,
                verbose=1,
                device='cuda',
                learning_rate=learning_rate
            )

        model.set_logger(new_logger)

        eval_env = Monitor(gym.make(self.env_name))
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=self.log_dir,
            log_path=self.log_dir,
            eval_freq=5000,
            deterministic=True,
            render=False,
            verbose=1
        )

        print("\n=== Iniciando Treinamento ===")
        start_time = time.time()
        
        model.learn(
            total_timesteps=episodes,
            progress_bar=True,
            callback=eval_callback,
            log_interval=100
        )
        
        elapsed_time = time.time() - start_time
        print("\n=== Treinamento Concluído ===")
        print(f"Tempo total: {elapsed_time/60:.1f} minutos")
        print(f"Steps/segundo: {episodes/elapsed_time:.1f}")
        print("============================\n")
        
        model.save(self.model_path)

    def eval(self):
        env = gym.make(self.env_name, render_mode="human")
        model = PPO.load(self.model_path)  # Corrigido: PPO ao invés de SAC
        obs, _ = env.reset()
        
        rewards = []
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            env.render()
            if done:
                obs, _ = env.reset()