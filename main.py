import argparse
import json

import agents

methods = {
    "SAC": agents.SACAgent,
    "PPO": agents.PPOAgent,  # Descomente o PPO
    # "DDPG": agents.DDPGAgent,
    # "TD3": agents.TD3Agent,
}

def main():
    parser = argparse.ArgumentParser(description="SSL Training Script")
    # Adicione os argumentos específicos do PPO para que possam ser lidos
    parser.add_argument("-m", "--method", type=str, help="Name of RL method")
    parser.add_argument("-n", "--name", type=str, help="Name of the agent")
    parser.add_argument("-e", "--env", type=str, help="Name of the environment")
    parser.add_argument("-v", "--version", type=str, help="Version of the agent")
    parser.add_argument("-t", "--train", action="store_true", help="Flag to indicate training mode")
    parser.add_argument("-j", "--json", type=str, help="Path to hyperparameters JSON file")

    # Argumentos que podem estar no JSON ou na linha de comando
    parser.add_argument("-ep", "--episodes", type=int, help="Number of timesteps for training")
    parser.add_argument("-mp", "--model_path", type=str, help="Path to a pre-trained model (optional)")
    parser.add_argument("-ne", "--num_envs", type=int, help="Number of environments to run in parallel")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size for training")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate for the optimizer")
    parser.add_argument("--n_steps", type=int, help="PPO: Number of steps per update")
    parser.add_argument("--n_epochs", type=int, help="PPO: Number of epochs per update")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, help="GAE Lambda")
    parser.add_argument("--clip_range", type=float, help="PPO clip range")

    args = parser.parse_args()

    # Transforma os argumentos em um dicionário de hiperparâmetros
    hyperparams = {}
    if args.json:
        with open(args.json, "r") as f:
            hyperparams.update(json.load(f))

    # Argumentos da linha de comando sobrescrevem os do JSON
    # Isso permite fazer testes rápidos mudando um só parâmetro
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    hyperparams.update(cli_args)

    # Validação simples
    required_args = ["method", "name", "env", "version"]
    if args.train:
        required_args.append("episodes")

    for arg in required_args:
        if arg not in hyperparams:
            parser.error(f"O argumento '{arg}' é obrigatório e não foi encontrado no JSON ou na linha de comando.")

    # Instancia o agente correto
    agent_class = methods.get(hyperparams["method"])
    if not agent_class:
        parser.error(f"Método '{hyperparams['method']}' não encontrado.")

    agent = agent_class(
        env_name=hyperparams["env"],
        agent_name=hyperparams["name"],
        agent_version=hyperparams["version"],
        # LINHA CRÍTICA: Passa o valor para o construtor do agente
        max_episode_steps=hyperparams.get("max_episode_steps", 1200)
    )

    print("Iniciando com os seguintes hiperparâmetros:")
    print(json.dumps(hyperparams, indent=4))

    if args.train:
        # --- MUDANÇA PRINCIPAL AQUI ---
        # Passa todos os hiperparâmetros de uma vez para o método train
        agent.train(**hyperparams)

    agent.eval()

if __name__ == "__main__":
    main()
