import gymnasium as gym
import numpy as np
# Importação chave do pacote de contribuições
from stable_baselines3_contrib import DDPG
# Importação para o Ruído de Ação (Action Noise), essencial para o DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from torch import nn
from torch.optim import Adam

from agents import AgentBase
from agents.callbacks import MetricsCallback


class DDPGAgent(AgentBase):
    def train(
        self,
        episodes=1000,
        model_path=None,
        num_envs=1,
        batch_size=256,
        learning_rate=1e-4,
        buffer_size=1_000_000, # Tamanho do Replay Buffer
        tau=0.005,             # Parâmetro de "soft update"
    ):
        # A criação do ambiente é a mesma, DDPG também é off-policy
        env = SubprocVecEnv([self._make_env for _ in range(num_envs)])
        env = VecMonitor(env)

        # Logger com TensorBoard (igual aos outros)
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        print(f"[DDPGAgent] Logger configured with log_dir: {self.log_dir}")

        if model_path:
            # Carrega um modelo DDPG existente
            model = DDPG.load(path=model_path, env=env, device="cuda")
        else:
            # --- CRIAÇÃO DO MODELO DDPG ---

            # O DDPG precisa de ruído para explorar. NormalActionNoise é uma boa escolha.
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            model = DDPG(
                "MlpPolicy",
                env,
                action_noise=action_noise,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                tau=tau,
                gamma=0.99,
                policy_kwargs=dict(
                    net_arch=[256, 256, 256],
                    activation_fn=nn.ReLU,
                ),
                verbose=0,
                device="cuda",
            )

        model.set_logger(new_logger)
        print("[DDPGAgent] Logger set for model")

        # Callbacks são os mesmos, são agnósticos ao algoritmo
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
        print("[DDPGAgent] MetricsCallback created")

        callbacks = [eval_callback, metrics_callback]

        model.learn(total_timesteps=episodes, progress_bar=True, callback=callbacks)
        model.save(self.model_path)

    def eval(self):
        # O método de avaliação é praticamente idêntico, só muda o `load`
        env = gym.make(self.env_name, render_mode="human")
        model = DDPG.load(self.model_path) # Carrega o modelo DDPG
        obs, _ = env.reset()

        rewards = []
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            # env.render() # O render já é chamado no step do seu ambiente
            if done:
                obs, _ = env.reset()
