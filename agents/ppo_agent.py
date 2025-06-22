import gymnasium as gym
from stable_baselines3 import PPO  # Trocamos SAC por PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from torch import nn
from torch.optim import Adam

from agents import AgentBase
from agents.callbacks import MetricsCallback


class PPOAgent(AgentBase):
    def train(
        self,
        episodes=1000000,
        model_path=None,
        num_envs=4,          # PPO se beneficia muito de ambientes paralelos
        batch_size=64,       # Minibatch size para cada época de atualização
        learning_rate=3e-4,  # Taxa de aprendizado comum para PPO
        n_steps=2048,        # Tamanho do rollout por ambiente antes da atualização
        n_epochs=10,         # Número de épocas de otimização por rollout
        gamma=0.99,          # Fator de desconto
    ):
        # Criar ambiente de treinamento (igual ao SAC)
        # PPO funciona muito bem com múltiplos ambientes em paralelo
        env = SubprocVecEnv([self._make_env for _ in range(num_envs)])
        env = VecMonitor(env)

        # Configurar logger (igual ao SAC)
        new_logger = configure(self.log_dir, ["stdout", "tensorboard"])
        print(f"[PPOAgent] Logger configured with log_dir: {self.log_dir}")

        if model_path:
            # Carrega um modelo PPO existente
            model = PPO.load(path=model_path, env=env, device="cuda")
        else:
            # Cria um novo modelo PPO
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=dict(
                    net_arch=[256, 256, 256],
                    activation_fn=nn.ReLU,
                    optimizer_class=Adam,
                ),
                n_steps=n_steps,              # Parâmetro chave do PPO
                batch_size=batch_size,        # Parâmetro chave do PPO
                n_epochs=n_epochs,            # Parâmetro chave do PPO
                gamma=gamma,                  # Fator de desconto
                learning_rate=learning_rate,
                verbose=0,
                device="cuda",
                ent_coef=0.01,                # Coeficiente de entropia (ajuste fino)
                gae_lambda=0.95,              # Fator GAE (Advantage Estimation)
                clip_range=0.2,               # Clipping do PPO
            )

        model.set_logger(new_logger)
        print("[PPOAgent] Logger set for model")

        # Callbacks são os mesmos, são agnósticos ao algoritmo
        eval_env = DummyVecEnv([lambda: Monitor(gym.make(self.env_name))])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.log_dir,
            log_path=self.log_dir,
            eval_freq=max(n_steps * 2 // num_envs, 1024), # Avaliar a cada ~2 rollouts
            deterministic=True,
            render=False,
        )

        metrics_callback = MetricsCallback(self.log_dir)
        print("[PPOAgent] MetricsCallback created")

        callbacks = [eval_callback, metrics_callback]

        model.learn(total_timesteps=episodes, progress_bar=True, callback=callbacks)
        model.save(self.model_path)

    def eval(self):
        # O método de avaliação é praticamente idêntico, só muda o `load`
        env = gym.make(self.env_name, render_mode="human")
        model = PPO.load(self.model_path) # Carrega o modelo PPO
        obs, _ = env.reset()

        rewards = []
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            env.render()
            if done:
                obs, _ = env.reset()
