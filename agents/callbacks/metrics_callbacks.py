import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import json
import os
from datetime import datetime

class MetricsCallback(BaseCallback):
    """
    Callback customizado para coletar e logar métricas detalhadas
    de cada episódio no TensorBoard e em um arquivo JSON.
    Totalmente alinhado com as métricas geradas por SSLELReachToPose.
    """
    def __init__(self, log_dir, save_every=100, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_every = save_every # Salva o JSON a cada 'save_every' episódios
        self.episode_num = 0 # Contador para o número de episódios processados

        print(f"[MetricsCallback] Initializing with log_dir: {log_dir}")

        # Dicionário para armazenar o histórico completo de métricas por episódio
        # AGORA INCLUI TODAS AS NOVAS MÉTRICAS!
        self.episode_metrics_history = {
            'success': [], # Boolean
            'out_of_bounds': [], # Boolean
            'timeout': [], # Boolean
            'initial_distance_pos': [], # float
            'initial_distance_angle_rad': [], # float
            'final_distance_pos': [], # float
            'final_theta_error_deg': [], # float
            'steps_to_goal': [], # int
            'path_length': [], # float
            'path_efficiency': [], # float
            'avg_linear_velocity': [], # float
            'avg_angular_velocity': [], # float
            'avg_linear_acceleration': [], # float
            'avg_angular_acceleration': [], # float
            'linear_jerk': [], # float
            'angular_jerk': [], # float
            'total_direction_changes': [], # int
            'mean_action_magnitude': [], # float
            'curriculum_phase': [], # float
            'episode_reward_total': [], # float (recompensa total do episódio)
            'reward_pos_step': [], # float (recompensa de posição por passo)
            'reward_orientation_step': [], # float (recompensa de orientação por passo)
            'action_penalty_step': [], # float (penalidade de ação por passo)
            'close_proximity_bonus_step': [], # float (bônus de proximidade por passo)
            'final_goal_bonus': [] # float (bônus final de gol)
        }

        self.metrics_dir = os.path.join(log_dir, 'custom_metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        print(f"[MetricsCallback] Created metrics directory: {self.metrics_dir}")

        self.metrics_file = os.path.join(
            self.metrics_dir,
            f'episode_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        print(f"[MetricsCallback] Will save metrics to: {self.metrics_file}")

    def _on_step(self) -> bool:
        """
        Este método é chamado a cada passo do ambiente.
        Aqui coletamos as informações de episódios completos quando eles terminam.
        """
        infos = self.locals.get("infos", [])

        for info in infos:
            # Verifica se o episódio terminou em algum dos ambientes paralelos
            # e se ele contém as métricas que esperamos (colocadas pelo ambiente)
            if "episode_metrics" in info: # 'episode_metrics' é o dicionário que o ambiente passa no info
                self.episode_num += 1
                metrics = info["episode_metrics"]

                # Coleta e converte para tipos Python nativos para JSON e TensorBoard
                # Garante que booleanos e np.float32 sejam convertidos corretamente.
                self.episode_metrics_history['success'].append(bool(metrics.get('success', False)))
                self.episode_metrics_history['out_of_bounds'].append(bool(metrics.get('out_of_bounds', False)))
                self.episode_metrics_history['timeout'].append(bool(metrics.get('timeout', False)))
                self.episode_metrics_history['initial_distance_pos'].append(float(metrics.get('initial_distance_pos', 0.0)))
                self.episode_metrics_history['initial_distance_angle_rad'].append(float(metrics.get('initial_distance_angle_rad', 0.0)))
                self.episode_metrics_history['final_distance_pos'].append(float(metrics.get('final_distance_pos', 0.0)))
                self.episode_metrics_history['final_theta_error_deg'].append(float(metrics.get('final_theta_error_deg', 0.0)))
                self.episode_metrics_history['steps_to_goal'].append(int(metrics.get('steps_to_goal', 0)))
                self.episode_metrics_history['path_length'].append(float(metrics.get('path_length', 0.0)))
                self.episode_metrics_history['path_efficiency'].append(float(metrics.get('path_efficiency', 0.0)))
                self.episode_metrics_history['avg_linear_velocity'].append(float(metrics.get('avg_linear_velocity', 0.0)))
                self.episode_metrics_history['avg_angular_velocity'].append(float(metrics.get('avg_angular_velocity', 0.0)))
                self.episode_metrics_history['avg_linear_acceleration'].append(float(metrics.get('avg_linear_acceleration', 0.0)))
                self.episode_metrics_history['avg_angular_acceleration'].append(float(metrics.get('avg_angular_acceleration', 0.0)))
                self.episode_metrics_history['linear_jerk'].append(float(metrics.get('linear_jerk', 0.0)))
                self.episode_metrics_history['angular_jerk'].append(float(metrics.get('angular_jerk', 0.0)))
                self.episode_metrics_history['total_direction_changes'].append(int(metrics.get('total_direction_changes', 0)))
                self.episode_metrics_history['mean_action_magnitude'].append(float(metrics.get('mean_action_magnitude', 0.0)))
                self.episode_metrics_history['curriculum_phase'].append(float(metrics.get('curriculum_phase', 1.0)))
                self.episode_metrics_history['episode_reward_total'].append(float(metrics.get('episode_reward_total', 0.0)))
                self.episode_metrics_history['reward_pos_step'].append(float(metrics.get('reward_pos_step', 0.0)))
                self.episode_metrics_history['reward_orientation_step'].append(float(metrics.get('reward_orientation_step', 0.0)))
                self.episode_metrics_history['action_penalty_step'].append(float(metrics.get('action_penalty_step', 0.0)))
                self.episode_metrics_history['close_proximity_bonus_step'].append(float(metrics.get('close_proximity_bonus_step', 0.0)))
                self.episode_metrics_history['final_goal_bonus'].append(float(metrics.get('final_goal_bonus', 0.0)))

                # Registra as métricas do episódio individualmente no TensorBoard
                if self.logger is not None:
                    self.logger.record('episode/success', bool(metrics.get('success', False)))
                    self.logger.record('episode/out_of_bounds', bool(metrics.get('out_of_bounds', False)))
                    self.logger.record('episode/timeout', bool(metrics.get('timeout', False)))
                    self.logger.record('episode/initial_distance_pos', float(metrics.get('initial_distance_pos', 0.0)))
                    self.logger.record('episode/initial_distance_angle_rad', float(metrics.get('initial_distance_angle_rad', 0.0)))
                    self.logger.record('episode/final_distance_pos', float(metrics.get('final_distance_pos', 0.0)))
                    self.logger.record('episode/final_theta_error_deg', float(metrics.get('final_theta_error_deg', 0.0)))
                    self.logger.record('episode/steps_to_goal', int(metrics.get('steps_to_goal', 0)))
                    self.logger.record('episode/path_length', float(metrics.get('path_length', 0.0)))
                    self.logger.record('episode/path_efficiency', float(metrics.get('path_efficiency', 0.0)))
                    self.logger.record('episode/avg_linear_velocity', float(metrics.get('avg_linear_velocity', 0.0)))
                    self.logger.record('episode/avg_angular_velocity', float(metrics.get('avg_angular_velocity', 0.0)))
                    self.logger.record('episode/avg_linear_acceleration', float(metrics.get('avg_linear_acceleration', 0.0)))
                    self.logger.record('episode/avg_angular_acceleration', float(metrics.get('avg_angular_acceleration', 0.0)))
                    self.logger.record('episode/linear_jerk', float(metrics.get('linear_jerk', 0.0)))
                    self.logger.record('episode/angular_jerk', float(metrics.get('angular_jerk', 0.0)))
                    self.logger.record('episode/total_direction_changes', int(metrics.get('total_direction_changes', 0)))
                    self.logger.record('episode/mean_action_magnitude', float(metrics.get('mean_action_magnitude', 0.0)))
                    self.logger.record('episode/curriculum_phase', float(metrics.get('curriculum_phase', 1.0)))
                    self.logger.record('episode/episode_reward_total', float(metrics.get('episode_reward_total', 0.0)))
                    self.logger.record('episode/reward_pos_step', float(metrics.get('reward_pos_step', 0.0)))
                    self.logger.record('episode/reward_orientation_step', float(metrics.get('reward_orientation_step', 0.0)))
                    self.logger.record('episode/action_penalty_step', float(metrics.get('action_penalty_step', 0.0)))
                    self.logger.record('episode/close_proximity_bonus_step', float(metrics.get('close_proximity_bonus_step', 0.0)))
                    self.logger.record('episode/final_goal_bonus', float(metrics.get('final_goal_bonus', 0.0)))

                # Salva o histórico completo no JSON periodicamente
                if self.episode_num % self.save_every == 0 and self.episode_num > 0:
                    self._save_metrics()

        # O TensorBoard dump é gerenciado pelo logger do modelo (SAC) automaticamente
        # ou pela chamada a _log_rolling_averages_to_tensorboard().
        # self.logger.dump(self.num_timesteps) é chamado no _log_rolling_averages_to_tensorboard.
        # Precisamos garantir que _log_rolling_averages_to_tensorboard seja chamado
        # em um intervalo razoável.
        # Para SAC (off-policy), _on_step é chamado em cada passo.
        # É melhor ter uma frequência para o log de médias móveis para não sobrecarregar o log.
        # A EvalCallback já faz um dump em eval_freq.
        # Podemos chamar _log_rolling_averages_to_tensorboard em um intervalo de timesteps.
        if self.num_timesteps % 1000 == 0: # Log rolling averages a cada 1000 timesteps
            self._log_rolling_averages_to_tensorboard()

        return True

    def _on_rollout_end(self) -> None:
        """
        Este método é chamado ao final de cada rollout. Para algoritmos off-policy como SAC,
        a coleta de métricas por episódio já é feita em _on_step quando done é True.
        Este método não é usado para agregação de métricas de episódio, mas pode ser usado
        para logs em intervalos de rollout. No nosso caso, _on_step já faz o trabalho.
        """
        pass # Não precisamos de lógica aqui, _on_step já gerencia.


    def _on_training_end(self) -> None:
        """
        Chamado uma vez no final do treinamento.
        Garante que todas as métricas sejam salvas.
        """
        print("[MetricsCallback] Training ended. Saving final metrics.")
        self._save_metrics()

    def _save_metrics(self):
        """
        Salva o histórico completo de métricas em um arquivo JSON.
        """
        # A estrutura aqui é para salvar o histórico de TODOS os episódios.
        # Se você quiser estatísticas agregadas (médias, desvios) do TOTAL do treinamento,
        # você pode calcular aqui antes de salvar.
        try:
            # Calcule médias finais de todas as métricas para a seção 'stats'
            stats = {
                'total_episodes': len(self.episode_metrics_history['episode_reward_total']),
                'mean_reward_total': float(np.mean(self.episode_metrics_history['episode_reward_total'])) if self.episode_metrics_history['episode_reward_total'] else 0.0,
                'std_reward_total': float(np.std(self.episode_metrics_history['episode_reward_total'])) if self.episode_metrics_history['episode_reward_total'] else 0.0,
                'mean_success_rate': float(np.mean(self.episode_metrics_history['success'])) if self.episode_metrics_history['success'] else 0.0,
                'mean_final_distance_pos': float(np.mean(self.episode_metrics_history['final_distance_pos'])) if self.episode_metrics_history['final_distance_pos'] else 0.0,
                'mean_final_theta_error_deg': float(np.mean(self.episode_metrics_history['final_theta_error_deg'])) if self.episode_metrics_history['final_theta_error_deg'] else 0.0,
                'mean_steps_to_goal': float(np.mean(self.episode_metrics_history['steps_to_goal'])) if self.episode_metrics_history['steps_to_goal'] else 0.0,
                'mean_path_efficiency': float(np.mean(self.episode_metrics_history['path_efficiency'])) if self.episode_metrics_history['path_efficiency'] else 0.0,
                'mean_avg_linear_velocity': float(np.mean(self.episode_metrics_history['avg_linear_velocity'])) if self.episode_metrics_history['avg_linear_velocity'] else 0.0,
                'mean_avg_angular_velocity': float(np.mean(self.episode_metrics_history['avg_angular_velocity'])) if self.episode_metrics_history['avg_angular_velocity'] else 0.0,
                'mean_linear_jerk': float(np.mean(self.episode_metrics_history['linear_jerk'])) if self.episode_metrics_history['linear_jerk'] else 0.0,
                'mean_angular_jerk': float(np.mean(self.episode_metrics_history['angular_jerk'])) if self.episode_metrics_history['angular_jerk'] else 0.0,
                'mean_total_direction_changes': float(np.mean(self.episode_metrics_history['total_direction_changes'])) if self.episode_metrics_history['total_direction_changes'] else 0.0,
                'mean_mean_action_magnitude': float(np.mean(self.episode_metrics_history['mean_action_magnitude'])) if self.episode_metrics_history['mean_action_magnitude'] else 0.0,
                'mean_curriculum_phase': float(np.mean(self.episode_metrics_history['curriculum_phase'])) if self.episode_metrics_history['curriculum_phase'] else 0.0,
                # Inclua também os históricos completos, se desejar (podem ser grandes para o JSON)
                'history': self.episode_metrics_history
            }

            with open(self.metrics_file, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"[MetricsCallback] Metrics history saved to {self.metrics_file}")
        except Exception as e:
            print(f"[MetricsCallback] Error saving metrics to JSON: {e}")

    def _log_rolling_averages_to_tensorboard(self):
        """
        Calcula e loga médias móveis de métricas no TensorBoard.
        Isso ajuda a visualizar a tendência de aprendizado.
        """
        if self.logger is not None:
            # Usar uma janela de 100 episódios para calcular médias móveis
            window_size = min(100, len(self.episode_metrics_history['episode_reward_total']))

            if window_size > 0:
                # Calcular médias móveis para todas as métricas relevantes
                for key in self.episode_metrics_history:
                    if len(self.episode_metrics_history[key]) >= window_size: # Garante que há dados suficientes
                        # Evita calcular média de booleanos diretamente
                        if isinstance(self.episode_metrics_history[key][0], bool):
                            # Para booleanos, a média é a taxa de ocorrência (e.g., success_rate)
                            self.logger.record(f'metrics/rolling_mean_{key}', np.mean(self.episode_metrics_history[key][-window_size:]))
                        elif isinstance(self.episode_metrics_history[key][0], (int, float)):
                            self.logger.record(f'metrics/rolling_mean_{key}', np.mean(self.episode_metrics_history[key][-window_size:]))
                        # Se for outra coisa (como lista dentro de lista), não tente a média simples.

                # Força o dump do logger para que as métricas apareçam no TensorBoard
                self.logger.dump(self.num_timesteps)
