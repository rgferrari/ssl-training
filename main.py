import argparse
import json

import agents

methods = {
    "SAC": agents.SACAgent,
    "PPO": agents.PPOAgent
}

def main():
    parser = argparse.ArgumentParser(description="SSL Training Script")
    parser.add_argument("-m", "--method", type=str, help="Name of RL method")
    parser.add_argument("-n", "--name", type=str, help="Name of the agent")
    parser.add_argument("-e", "--env", type=str, help="Name of the environment")
    parser.add_argument("-v", "--version", type=str, help="Version of the agent")
    parser.add_argument("-t", "--train", action="store_true", help="Flag to indicate training mode")
    parser.add_argument("-j", "--json", type=str, help="Path to hyperparameters JSON file")
    parser.add_argument("-ep", "--episodes", type=int, help="Number of episodes for training")
    parser.add_argument("-mp", "--model_path", type=str, help="Path to a pre-trained model (optional)")
    parser.add_argument("-ne", "--num_envs", type=int, help="Number of environments to run in parallel")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size for training")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate for the optimizer")
    args = parser.parse_args()
    

    # Fill empty args if json file is provided
    if args.json:
        with open(args.json, "r") as f:
            json_args = json.load(f)
        for key, value in json_args.items():
            if not getattr(args, key, None):
                setattr(args, key, value)

    required_args_training = [
        "method", 
        "name", 
        "env", 
        "version", 
        "episodes", 
        "num_envs", 
        "batch_size", 
        "learning_rate"
    ]

    required_args_eval = [
        "method", 
        "name", 
        "env", 
        "version"
    ]

    missing_args = [arg for arg in required_args_training if not getattr(args, arg, None)]
    if args.train:
        if missing_args:
            parser.error(f"The following arguments are required: {', '.join(missing_args)}")
            print("Hyperparameters:")
            for arg, value in vars(args).items():
                print(f"{arg}: {value}")
    else:
        missing_args = [arg for arg in required_args_eval if not getattr(args, arg, None)]
        if missing_args:
            parser.error(f"The following arguments are required: {', '.join(missing_args)}")
            print("Hyperparameters:")
            for arg, value in vars(args).items():
                print(f"{arg}: {value}")

    agent = methods[args.method](
        env_name=args.env,
        agent_name=args.name,
        agent_version=args.version
    )

    if args.train:
        agent.train(episodes=args.episodes, 
                    model_path=args.model_path, 
                    num_envs=args.num_envs, 
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate)
    agent.eval()

if __name__ == "__main__":
    main()
