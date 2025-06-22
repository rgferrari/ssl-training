import gymnasium as gym
import numpy as np
# Importação da biblioteca padrão do SB3
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from torch import nn
from torch.optim import Adam

from agents import AgentBase
from agents.callbacks import MetricsCallback


class TD3Agent(AgentBase):
    def train(
        self,
        episodes=1000,
        model_path=None,
        num_envs=1,
        batch_size=256,
        learning_rate=1e-4,
        buffer_size=1_000_000,
        tau=0.005,
        policy_delay=2, # Parâmetro chave do TD3
    ):
        # A criação do ambiente é a mesma, TD3 também é off-policy
        env = SubprocVecEnv([self._make_env for _ in range(num_envs)])
        env = VecMonitor(env)

        # Logger com TensorBoard (igual aos outros)
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        print(f"[TD3Agent] Logger configured with log_dir: {self.log_dir}")

        if model_path:
            # Carrega um modelo TD3 existente
            model = TD3.load(path=model_path, env=env, device="cuda")
        else:
            # --- CRIAÇÃO DO MODELO TD3 ---

            # Assim como o DDPG, o TD3 usa ruído de ação para exploração
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            model = TD3(
                "MlpPolicy",
                env,
                action_noise=action_noise,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                tau=tau,
                gamma=0.99,
                policy_delay=policy_delay, # Atraso na atualização da política
                policy_kwargs=dict(
                    net_arch=[256, 256, 256],
                    activation_fn=nn.ReLU,
                ),
                verbose=0,
                device="cuda",
            )

        model.set_logger(new_logger)
        print("[TD3Agent] Logger set for model")

        # Callbacks são os mesmos
        eval_env = DummyVecEnv([lambda: Monitor(gym.make(self.env_name))])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.log_dir,
            log_path=self.log_dir,
            eval_freq=5000,
            deterministic=True,
            render=False,
        )

        metrics_callback = MetricsCallback(self.log_dir)
        print("[TD3Agent] MetricsCallback created")

        callbacks = [eval_callback, metrics_callback]

        model.learn(total_timesteps=episodes, progress_bar=True, callback=callbacks)
        model.save(self.model_path)

    def eval(self):
        # O método de avaliação é praticamente idêntico
        env = gym.make(self.env_name, render_mode="human")
        model = TD3.load(self.model_path) # Carrega o modelo TD3
        obs, _ = env.reset()

        rewards = []
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            # env.render() é chamado no step do seu ambiente
            if done:
                obs, _ = env.reset()
