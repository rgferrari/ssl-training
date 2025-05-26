import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from torch import nn
from torch.optim import Adam

from agents import AgentBase
from agents.callbacks import MetricsCallback


class SACAgent(AgentBase):
    def train(
        self,
        episodes=1000,
        model_path=None,
        num_envs=1,
        batch_size=256,
        learning_rate=1e-4,
    ):
        # Criar ambiente de treinamento
        env = SubprocVecEnv([self._make_env for _ in range(num_envs)])
        env = VecMonitor(env)  # Adicionar monitor para coletar métricas

        # Configurar logger com TensorBoard
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        print(f"[SACAgent] Logger configured with log_dir: {self.log_dir}")  # Debug print

        if model_path:
            model = SAC.load(path=model_path, env=env, device="cuda")
        else:
            model = SAC(
                "MlpPolicy",
                env,
                policy_kwargs=dict(
                    net_arch=[256, 256, 256],
                    activation_fn=nn.ReLU,
                    optimizer_class=Adam,
                ),
                batch_size=batch_size,
                verbose=0,
                device="cuda",
                learning_rate=learning_rate,
                ent_coef="auto_0.1",
            )

        model.set_logger(new_logger)
        print("[SACAgent] Logger set for model")  # Debug print

        # Criar ambiente de avaliação
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
        print("[SACAgent] MetricsCallback created")  # Debug print
        
        callbacks = [eval_callback, metrics_callback]

        model.learn(total_timesteps=episodes, progress_bar=True, callback=callbacks)
        model.save(self.model_path)

    def eval(self):
        env = gym.make(self.env_name, render_mode="human")
        model = SAC.load(self.model_path)
        obs, _ = env.reset()

        rewards = []
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            env.render()
            if done:
                obs, _ = env.reset()