import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from torch import nn

# Supondo que AgentBase está em agents/__init__.py ou similar
from . import AgentBase
# Supondo que MetricsCallback está em agents/callbacks.py
from .callbacks import MetricsCallback


class PPOAgent(AgentBase):
    """
    Agente que encapsula o algoritmo PPO do stable-baselines3.
    """
    def train(self, **kwargs):
        """
        Treina o agente PPO usando hiperparâmetros passados via kwargs.
        """
        # Extrai os hiperparâmetros do kwargs com valores padrão
        total_timesteps = kwargs.get("total_timesteps",100_000)
        model_path = kwargs.get("model_path", None)
        num_envs = kwargs.get("num_envs", 16)
        learning_rate = kwargs.get("learning_rate", 3e-4)
        n_steps = kwargs.get("n_steps", 2048)
        batch_size = kwargs.get("batch_size", 64)
        n_epochs = kwargs.get("n_epochs", 10)
        gamma = kwargs.get("gamma", 0.99)
        gae_lambda = kwargs.get("gae_lambda", 0.95)
        clip_range = kwargs.get("clip_range", 0.2)

        # Criar ambiente de treinamento vetorizado
        env = SubprocVecEnv([self._make_env for _ in range(num_envs)])
        env = VecMonitor(env)

        # Configurar logger
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        print(f"[PPOAgent] Logger configurado em: {self.log_dir}")

        if model_path:
            model = PPO.load(path=model_path, env=env, device="cuda")
            print(f"[PPOAgent] Modelo carregado de {model_path}")
        else:
            print("[PPOAgent] Criando novo modelo PPO...")
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.ReLU),
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                verbose=0,
                device="cuda",
                learning_rate=learning_rate,
            )

        model.set_logger(new_logger)

        # Callbacks
        eval_env = DummyVecEnv([lambda: Monitor(self._make_env())])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.log_dir,
            log_path=self.log_dir,
            eval_freq=max(33000 // num_envs, 1), # Avalia a cada ~10000 passos globais
            deterministic=True,
            render=False,
        )
        metrics_callback = MetricsCallback(self.log_dir)
        callbacks = [eval_callback, metrics_callback]

        # Treinamento
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)
        model.save(self.model_path)
        print(f"[PPOAgent] Modelo salvo em {self.model_path}")

    def eval(self):
        env = gym.make(self.env_name, render_mode="human")
        model = PPO.load(self.model_path)
        obs, _ = env.reset()
        for _ in range(2000): # Aumentado para uma avaliação mais longa
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            if done:
                obs, _ = env.reset()
