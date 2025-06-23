import os

import gymnasium as gym

class AgentBase():
    def __init__(self, env_name, agent_name, agent_version, max_episode_steps):
        self.env_name = env_name
        self.agent_name = agent_name
        self.agent_version = agent_version

        self.log_dir = f"./logs/{self.agent_name}/{self.agent_name}-v_{self.agent_version}"
        self.model_path = f"./models/{self.agent_name}/{self.agent_name}-v_{self.agent_version}"
        self.max_episode_steps = max_episode_steps

        os.makedirs(self.log_dir, exist_ok=True)

    def _make_env(self):
        """Cria uma instância do ambiente, passando o max_episode_steps."""
        # AQUI é onde o valor é finalmente usado!
        env = gym.make(self.env_name, max_episode_steps=self.max_episode_steps)
        return env

    def train(self):
        raise NotImplementedError("Train method not implemented")

    def eval(self):
        raise NotImplementedError("Eval method not implemented")
