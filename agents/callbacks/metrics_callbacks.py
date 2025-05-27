import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import json
import os
from datetime import datetime

class MetricsCallback(BaseCallback):
    def __init__(self, log_dir, save_every=100, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_every = save_every
        print(f"[MetricsCallback] Initializing with log_dir: {log_dir}")
        
        self.episode_metrics = {
            'rewards': [],
            'success_rate': [],
            'out_of_bounds_rate': [],
            'timeout_rate': [],
            'final_distances': [],
            'steps_to_target': [],
            'path_efficiency': [],
            'avg_velocity': [],
            'avg_acceleration': [],
            'direction_changes': []
        }
        
        self.metrics_dir = os.path.join(log_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        print(f"[MetricsCallback] Created metrics directory: {self.metrics_dir}")
        
        self.metrics_file = os.path.join(
            self.metrics_dir, 
            f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        print(f"[MetricsCallback] Will save metrics to: {self.metrics_file}")

    def _on_step(self) -> bool:
        # 'infos' está disponível em self.locals
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_metrics" in info:
                metrics = info["episode_metrics"]
                # Converta para tipos nativos
                reward = float(metrics.get('episode_reward', 0))
                success = 1 if metrics.get('success', False) else 0
                out_of_bounds = 1 if metrics.get('out_of_bounds', False) else 0
                final_distance = float(metrics.get('final_distance', 0))
                timeout = 1 if metrics.get('timeout', False) else 0

                # Salve no histórico (para o JSON)
                self.episode_metrics['rewards'].append(reward)
                self.episode_metrics['success_rate'].append(success)
                self.episode_metrics['out_of_bounds_rate'].append(out_of_bounds)
                self.episode_metrics['timeout_rate'].append(timeout)
                self.episode_metrics['final_distances'].append(final_distance)
                self.episode_metrics['steps_to_target'].append(int(metrics.get('steps_to_target', 0)))
                self.episode_metrics['path_efficiency'].append(float(metrics.get('path_efficiency', 0)))
                self.episode_metrics['avg_velocity'].append(float(metrics.get('avg_velocity', 0)))
                self.episode_metrics['avg_acceleration'].append(float(metrics.get('avg_acceleration', 0)))
                self.episode_metrics['direction_changes'].append(int(metrics.get('direction_changes', 0)))

                # Salve cada valor individualmente no TensorBoard
                if self.logger is not None:
                    self.logger.record('metrics/episode_reward', reward)
                    self.logger.record('metrics/success', success)
                    self.logger.record('metrics/out_of_bounds', out_of_bounds)
                    self.logger.record('metrics/timeout', timeout)
                    self.logger.record('metrics/final_distance', final_distance)
                    self.logger.record('metrics/steps_to_target', int(metrics.get('steps_to_target', 0)))
                    self.logger.record('metrics/path_efficiency', float(metrics.get('path_efficiency', 0)))
                    self.logger.record('metrics/avg_velocity', float(metrics.get('avg_velocity', 0)))
                    self.logger.record('metrics/avg_acceleration', float(metrics.get('avg_acceleration', 0)))
                    self.logger.record('metrics/direction_changes', int(metrics.get('direction_changes', 0)))
        # Salva a cada N episódios
        if len(self.episode_metrics['rewards']) % self.save_every == 0 and len(self.episode_metrics['rewards']) > 0:
            self._save_metrics()
        return True

    def _on_rollout_end(self) -> None:
        print(f"[MetricsCallback] Rollout ended. Buffer size: {len(self.model.ep_info_buffer)}")
        
        # Coletar métricas do episódio atual
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'episode_metrics' in info:
                    metrics = info['episode_metrics']
                    print(f"[MetricsCallback] Processing metrics: {metrics}")
                    
                    # Converter valores numpy para Python nativo
                    self.episode_metrics['rewards'].append(float(metrics.get('episode_reward', 0)))
                    self.episode_metrics['success_rate'].append(1 if metrics.get('success', False) else 0)
                    self.episode_metrics['out_of_bounds_rate'].append(1 if metrics.get('out_of_bounds', False) else 0)
                    self.episode_metrics['timeout_rate'].append(1 if metrics.get('timeout', False) else 0)
                    self.episode_metrics['final_distances'].append(float(metrics.get('final_distance', 0)))
                    self.episode_metrics['steps_to_target'].append(int(metrics.get('steps_to_target', 0)))
                    self.episode_metrics['path_efficiency'].append(float(metrics.get('path_efficiency', 0)))
                    self.episode_metrics['avg_velocity'].append(float(metrics.get('avg_velocity', 0)))
                    self.episode_metrics['avg_acceleration'].append(float(metrics.get('avg_acceleration', 0)))
                    self.episode_metrics['direction_changes'].append(int(metrics.get('direction_changes', 0)))
            
            # Registrar no TensorBoard
            self._log_metrics_to_tensorboard()

    def _on_training_end(self) -> None:
        # Salva ao final do treinamento também
        self._save_metrics()

    def _save_metrics(self):
        stats = {
            'episodes': len(self.episode_metrics['rewards']),
            'rewards': self.episode_metrics['rewards'],  # Histórico completo
            'success_rate': self.episode_metrics['success_rate'],
            'out_of_bounds_rate': self.episode_metrics['out_of_bounds_rate'],
            'timeout_rate': self.episode_metrics['timeout_rate'],
            'final_distances': self.episode_metrics['final_distances'],
            'steps_to_target': self.episode_metrics['steps_to_target'],
            'path_efficiency': self.episode_metrics['path_efficiency'],
            'avg_velocity': self.episode_metrics['avg_velocity'],
            'avg_acceleration': self.episode_metrics['avg_acceleration'],
            'direction_changes': self.episode_metrics['direction_changes'],
            'mean_reward': float(np.mean(self.episode_metrics['rewards'])) if self.episode_metrics['rewards'] else 0.0,
            'std_reward': float(np.std(self.episode_metrics['rewards'])) if self.episode_metrics['rewards'] else 0.0,
            'success_rate': float(np.mean(self.episode_metrics['success_rate'])) if self.episode_metrics['success_rate'] else 0.0,
            'out_of_bounds_rate': float(np.mean(self.episode_metrics['out_of_bounds_rate'])) if self.episode_metrics['out_of_bounds_rate'] else 0.0,
            'timeout_rate': float(np.mean(self.episode_metrics['timeout_rate'])) if self.episode_metrics['timeout_rate'] else 0.0,
            'mean_final_distance': float(np.mean(self.episode_metrics['final_distances'])) if self.episode_metrics['final_distances'] else 0.0,
            'mean_steps_to_target': float(np.mean(self.episode_metrics['steps_to_target'])) if self.episode_metrics['steps_to_target'] else 0.0,
            'mean_path_efficiency': float(np.mean(self.episode_metrics['path_efficiency'])) if self.episode_metrics['path_efficiency'] else 0.0,
            'mean_velocity': float(np.mean(self.episode_metrics['avg_velocity'])) if self.episode_metrics['avg_velocity'] else 0.0,
            'mean_acceleration': float(np.mean(self.episode_metrics['avg_acceleration'])) if self.episode_metrics['avg_acceleration'] else 0.0,
            'mean_direction_changes': float(np.mean(self.episode_metrics['direction_changes'])) if self.episode_metrics['direction_changes'] else 0.0
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"[MetricsCallback] Metrics saved to {self.metrics_file}")

    def _log_metrics_to_tensorboard(self):
        if self.logger is not None:
            # Usar as últimas 100 métricas para calcular médias móveis
            window_size = min(100, len(self.episode_metrics['rewards']))
            
            if window_size > 0:
                # Calcular médias móveis
                recent_rewards = self.episode_metrics['rewards'][-window_size:]
                recent_success = self.episode_metrics['success_rate'][-window_size:]
                recent_out_of_bounds = self.episode_metrics['out_of_bounds_rate'][-window_size:]
                recent_timeout = self.episode_metrics['timeout_rate'][-window_size:]
                recent_distances = self.episode_metrics['final_distances'][-window_size:]
                recent_steps = self.episode_metrics['steps_to_target'][-window_size:]
                recent_efficiency = self.episode_metrics['path_efficiency'][-window_size:]
                recent_velocity = self.episode_metrics['avg_velocity'][-window_size:]
                recent_acceleration = self.episode_metrics['avg_acceleration'][-window_size:]
                recent_direction_changes = self.episode_metrics['direction_changes'][-window_size:]

                # Registrar métricas no TensorBoard
                self.logger.record('metrics/mean_reward', np.mean(recent_rewards))
                self.logger.record('metrics/success_rate', np.mean(recent_success))
                self.logger.record('metrics/out_of_bounds_rate', np.mean(recent_out_of_bounds))
                self.logger.record('metrics/timeout_rate', np.mean(recent_timeout))
                self.logger.record('metrics/mean_final_distance', np.mean(recent_distances))
                self.logger.record('metrics/mean_steps_to_target', np.mean(recent_steps))
                self.logger.record('metrics/mean_path_efficiency', np.mean(recent_efficiency))
                self.logger.record('metrics/mean_velocity', np.mean(recent_velocity))
                self.logger.record('metrics/mean_acceleration', np.mean(recent_acceleration))
                self.logger.record('metrics/mean_direction_changes', np.mean(recent_direction_changes))
                
                self.logger.dump(self.num_timesteps)
        else:
            print("[MetricsCallback] No logger available")  # Debug print
