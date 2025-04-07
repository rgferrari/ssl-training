## Setup

Create the local venv and install all dependencies by running ```make all```

## Creating an Agent

All agents should implement the class [AgentBase](/agents/agent_base.py).

All agents should be imported in [agents/\_\_init\_\_.py](agents/__init__.py).

The new agent should be included in the methods dict in [main](main.py), so it can be called as an argument.

## Creating an Env

All environments should implement the rSoccer class [SSLBaseEnv](https://github.com/robocin/rSoccer/blob/main/rsoccer_gym/ssl/ssl_gym_base.py).

All environments should be imported and registered into gymnasium in [playgrounds/\_\_init\_\_.py](playgrounds/__init__.py).

## Train

Fill all the arguments into the [hyperparameters.json](hyperparameters.json) file.

The following arguments are required:
```
"method": Name of RL method
"name": Name of the agent
"env": Name of the environment
"version": Version of the environment
"episodes": Number of episodes for training
"num_envs": Number of environments to run in parallel
"batch_size": Batch size for training 
"learning_rate": Learning rate for the optimizer
```

You can also add the argument ```"model_path"``` if you want to finetune some existing model.

Check the [main](main.py) to see the other optional arguments.

Run ```make train``` to start the training.

In another terminal, run ```tensorboard --logdir=logs``` to start tensorboard, so you can keep up with the training charts.

Run ```make eval``` to run only the eval and watch training result.


## Available Agents
- [SacAgent](agents/sac_agent.py)

## Available Playgrounds
- [SSLReachBall-v0](playgrounds/reach_ball.py)
