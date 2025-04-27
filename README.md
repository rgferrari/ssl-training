# SSL Training

A framework for training Reinforcement Learning agents in simulated robot soccer environments.

---

## Setup

Set up your environment and install all dependencies:

```bash
make all
```

This will create a virtual environment locally and install everything listed in the `setup.py`.

---

## Creating a New Agent

To add a new RL agent:

1. **Implement** the agent by inheriting from [`AgentBase`](agents/agent_base.py).
2. **Import** your new agent class in [`agents/__init__.py`](agents/__init__.py).
3. **Register** the agent in the `methods` dictionary inside [`main.py`](main.py) so it can be selected from the CLI.

---

## Creating a New Environment

To define a new simulation environment:

1. **Implement** your environment by inheriting from [`SSLBaseEnv`](https://github.com/robocin/rSoccer/blob/main/rsoccer_gym/ssl/ssl_gym_base.py).
2. **Import and register** it in [`playgrounds/__init__.py`](playgrounds/__init__.py) using `gymnasium`.

---

## Training

### 1. Configure Hyperparameters

Edit [`hyperparameters.json`](hyperparameters.json) with the following required fields:

```json
{
  "method": "Name of RL method",
  "name": "Name of the agent",
  "env": "Name of the environment",
  "version": "Environment version",
  "episodes": "Number of episodes for training"
  "num_envs": "Number of environments to run in parallel"
  "batch_size": "Batch size for training" 
  "learning_rate": "Learning rate for the optimizer"
}
```

**Optional:**  
Add `"model_path"` to fine-tune from an existing model.

For additional configuration options, check [`main.py`](main.py).

---

### 2. Start Training

```bash
make train
```

### 3. Monitor with TensorBoard

In a separate terminal:

```bash
tensorboard --logdir=logs
```

This will show you real-time training metrics and performance graphs.

---

### 4. Run Evaluation Only

To evaluate the model:

```bash
make eval
```

---

## Outputs

- **Logs:** `logs/<agent_name>/<agent_name>-v_<version>/`
- **Models:** `models/<agent_name>/<agent_name>-v_<version>/`

---

## Available Agents

- [`SacAgent`](agents/sac_agent.py)

---

## Available Environments

- [`SSLReachBall-v0`](playgrounds/reach_ball.py)
