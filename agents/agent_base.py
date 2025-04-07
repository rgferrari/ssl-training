import os

import gymnasium as gym

class AgentBase():
    def __init__(self, env_name, agent_name, agent_version):
        self.env_name = env_name
        self.agent_name = agent_name
        self.agent_version = agent_version

        self.log_dir = f"./logs/{self.agent_name}/{self.agent_name}-v_{self.agent_version}"
        self.model_path = f"./models/{self.agent_name}/{self.agent_name}-v_{self.agent_version}"

        os.makedirs(self.log_dir, exist_ok=True)

    def _make_env(self, render_mode=None):
        return gym.make(self.env_name, render_mode=render_mode)
    
    def train(self):
        raise NotImplementedError("Train method not implemented")
    
    def eval(self):
        raise NotImplementedError("Eval method not implemented")