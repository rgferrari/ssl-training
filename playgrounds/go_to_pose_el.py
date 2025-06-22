import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import Optional, Tuple

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render.utils import COLORS
from playgrounds.Env.SSLELSim import SSL_EL_Env
import pygame


@dataclass
class Pose:
    x: float
    y: float
    theta: float  # em radianos

    def to_array(self):
        return np.array([self.x, self.y, self.theta])

    def distance_to(self, other: 'Pose') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def angular_distance_to(self, other: 'Pose') -> float:
        """Retorna distância angular em radianos (0 a pi)"""
        diff = abs(self.theta - other.theta)
        return min(diff, 2*np.pi - diff)


class SSLGoToPoseEnv(SSL_EL_Env):
    """Ambiente para treinar a skill fundamental de ir até uma pose específica

    Description:
        O robô deve navegar até uma pose alvo (x, y, θ) de forma eficiente.
        Este é o building block fundamental para todas as outras skills.

    Observation:
        Type: Box(12,)
        Normalized Bounds to [-1.2, 1.2]
        Num     Observation normalized
        0       Robot's X position
        1       Robot's Y position
        2       sin(Robot's Angle)
        3       cos(Robot's Angle)
        4       Error X (robot's frame)
        5       Error Y (robot's frame)
        6       sin(Error Angle)
        7       cos(Error Angle)
        8       Target's X position
        9       Target's Y position
        10      sin(Target's Angle)
        11      cos(Target's Angle)

    Actions:
        Type: Box(3,)
        Value Range: [-1, 1]
        Num     Action
        0       V_X (no referencial do robô)
        1       V_y (no referencial do robô)
        2       V_theta

    Reward:
        - Sparse reward only at the end:
          - Success: 500 * time_factor * distance_factor
          - Timeout: penalty based on remaining distances
          - Out of bounds: -10

    Starting State:
        - Poses inicial e alvo são aleatórias
        - Distância mínima entre elas configurable

    Episode Termination:
        - Quando atinge a pose alvo (sucesso)
        - Quando sai do campo
        - Após max_steps (falha)
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode, n_robots_blue=1, n_robots_yellow=0)

        # Configuração dos espaços
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        # ATUALIZADO: Espaço de observação com 12 elementos
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(12,),
            dtype=np.float32,
        )

        self.max_v = 2.5
        self.max_w = 10.0

        # Parâmetros do ambiente
        self.max_steps = 1200
        self.position_tolerance = 0.1
        self.angular_tolerance = 0.1
        # Distância máxima para normalização correta
        self.max_field_dist = np.sqrt(self.field.length**2 + self.field.width**2)

        # Estado interno
        self.target_pose: Optional[Pose] = None
        self.initial_distance: Optional[float] = None
        self.initial_angular_distance: Optional[float] = None

        # Métricas do episódio
        self.episode_steps = 0
        self.episode_reward_total = 0.0
        self.path_length = 0.0
        self.previous_position = None
        self.velocities_linear = []
        self.velocities_angular = []
        self.accelerations_linear = []
        self.accelerations_angular = []

    def reset(self, *, seed=None, options=None):
        """Reset com tracking de métricas"""
        # Reset das métricas
        self.episode_steps = 0
        self.episode_reward_total = 0.0
        self.path_length = 0.0
        self.previous_position = None
        self.velocities_linear = []
        self.velocities_angular = []
        self.accelerations_linear = []
        self.accelerations_angular = []
        self.previous_distance = None # Reseta a distância no início de cada episódio

        # Reset padrão
        return super().reset(seed=seed, options=options)

    def step(self, action):
        """Step com tracking de métricas"""
        # Step padrão
        obs, reward, terminated, truncated, info = super().step(action)

        # Atualizar métricas
        self.episode_steps += 1
        self.episode_reward_total += reward

        # Tracking de movimento
        robot = self.frame.robots_blue[0]
        current_position = np.array([robot.x, robot.y])

        # Calcular path length
        if self.previous_position is not None:
            self.path_length += np.linalg.norm(current_position - self.previous_position)
        self.previous_position = current_position

        # Tracking de velocidades
        linear_vel = np.sqrt(robot.v_x**2 + robot.v_y**2)
        angular_vel = abs(robot.v_theta)
        self.velocities_linear.append(linear_vel)
        self.velocities_angular.append(angular_vel)

        # Se terminou, calcular todas as métricas
        if terminated or truncated:
            info['episode_metrics'] = self._calculate_episode_metrics()

        return obs, reward, terminated, truncated, info

    def _calculate_episode_metrics(self):
        """Calcula todas as métricas do episódio finalizado"""
        robot = self.frame.robots_blue[0]
        current_pose = Pose(robot.x, robot.y, np.deg2rad(robot.theta))

        # Distâncias finais
        final_distance_pos = current_pose.distance_to(self.target_pose)
        final_angular_distance = current_pose.angular_distance_to(self.target_pose)

        # Status do episódio
        success = (final_distance_pos < self.position_tolerance and
                   final_angular_distance < self.angular_tolerance)
        out_of_bounds = self._is_out_of_bounds(robot)
        timeout = self.episode_steps >= self.max_steps

        # Eficiência do caminho
        straight_line_distance = self.initial_distance
        path_efficiency = straight_line_distance / self.path_length if self.path_length > 0 else 0.0

        # Velocidades médias
        avg_linear_velocity = np.mean(self.velocities_linear) if self.velocities_linear else 0.0
        avg_angular_velocity = np.mean(self.velocities_angular) if self.velocities_angular else 0.0

        # Acelerações
        if len(self.velocities_linear) > 1:
            linear_accels = np.diff(self.velocities_linear)
            angular_accels = np.diff(self.velocities_angular)
            avg_linear_acceleration = np.mean(np.abs(linear_accels))
            avg_angular_acceleration = np.mean(np.abs(angular_accels))

            # Jerk
            if len(linear_accels) > 1:
                linear_jerk = np.mean(np.abs(np.diff(linear_accels)))
                angular_jerk = np.mean(np.abs(np.diff(angular_accels)))
            else:
                linear_jerk, angular_jerk = 0.0, 0.0
        else:
            avg_linear_acceleration, avg_angular_acceleration = 0.0, 0.0
            linear_jerk, angular_jerk = 0.0, 0.0

        metrics = {
            'success': success,
            'out_of_bounds': out_of_bounds,
            'timeout': timeout,
            'initial_distance_pos': self.initial_distance,
            'initial_distance_angle_rad': self.initial_angular_distance,
            'final_distance_pos': final_distance_pos,
            'final_theta_error_deg': np.rad2deg(final_angular_distance),
            'steps_to_goal': self.episode_steps if success else self.max_steps,
            'path_length': self.path_length,
            'path_efficiency': path_efficiency,
            'avg_linear_velocity': avg_linear_velocity,
            'avg_angular_velocity': avg_angular_velocity,
            'avg_linear_acceleration': avg_linear_acceleration,
            'avg_angular_acceleration': avg_angular_acceleration,
            'linear_jerk': linear_jerk,
            'angular_jerk': angular_jerk,
            'episode_reward_total': self.episode_reward_total,
            'final_goal_bonus': self.episode_reward_total if success else 0.0
        }
        return metrics

    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Ações já estão no referencial do robô (Vx, Vy, Vw)
        # Apenas denormaliza e clipa
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w

        # Converte para referencial global para o simulador
        v_x_global = v_x * np.cos(angle) - v_y * np.sin(angle)
        v_y_global = v_x * np.sin(angle) + v_y * np.cos(angle)

        # Clip pelo módulo da velocidade linear
        v_norm = np.linalg.norm([v_x_global, v_y_global])
        if v_norm > self.max_v:
            v_x_global = (v_x_global / v_norm) * self.max_v
            v_y_global = (v_y_global / v_norm) * self.max_v

        return v_x_global, v_y_global, v_theta

    def _get_commands(self, actions):
        """Converte ações em comandos para o robô"""
        commands = []
        robot_angle_rad = np.deg2rad(self.frame.robots_blue[0].theta)
        v_x, v_y, v_theta = self.convert_actions(actions, robot_angle_rad)
        cmd = Robot(yellow=False, id=0, v_x=v_x, v_y=v_y, v_theta=v_theta)
        commands.append(cmd)
        return commands

    def _frame_to_observations(self):
        """Converte o estado do ambiente em observações com base em erro e alvo absoluto."""
        robot = self.frame.robots_blue[0]

        # --- Poses em radianos para cálculos ---
        robot_theta_rad = np.deg2rad(robot.theta)
        target_theta_rad = self.target_pose.theta

        # --- Cálculo do Erro ---
        # Erro de posição no referencial do mundo
        dx_world = self.target_pose.x - robot.x
        dy_world = self.target_pose.y - robot.y

        # Rotação do erro para o referencial do robô
        dx_robot = dx_world * np.cos(robot_theta_rad) + dy_world * np.sin(robot_theta_rad)
        dy_robot = -dx_world * np.sin(robot_theta_rad) + dy_world * np.cos(robot_theta_rad)

        # Erro de orientação (já em radianos)
        dtheta = target_theta_rad - robot_theta_rad
        # Normaliza dtheta para o intervalo [-pi, pi]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        # --- Normalização ---
        # Usa a diagonal do campo para normalizar as distâncias de erro
        dx_robot_norm = dx_robot / self.max_field_dist
        dy_robot_norm = dy_robot / self.max_field_dist

        # --- Vetor de Observações Final ---
        obs = [
            # 1. Estado do robô (4 valores)
            self.norm_pos(robot.x),
            self.norm_pos(robot.y),
            np.sin(robot_theta_rad),
            np.cos(robot_theta_rad),

            # 2. Erros normalizados (4 valores)
            dx_robot_norm,
            dy_robot_norm,
            np.sin(dtheta),
            np.cos(dtheta),

            # 3. Pose absoluta do alvo (4 valores)
            self.norm_pos(self.target_pose.x),
            self.norm_pos(self.target_pose.y),
            np.sin(target_theta_rad),
            np.cos(target_theta_rad)
        ]

        return np.array(obs, dtype=np.float32)


    def _calculate_reward_and_done(self):
        """Calcula a recompensa e condição de término com shaping."""
        robot = self.frame.robots_blue[0]
        current_pose = Pose(robot.x, robot.y, np.deg2rad(robot.theta))

        # Distâncias atuais
        position_distance = current_pose.distance_to(self.target_pose)
        angular_distance = current_pose.angular_distance_to(self.target_pose)

        reward = 0
        done = False

        # --- Lógica de Término ---

        # 1. Saiu dos limites
        if self._is_out_of_bounds(robot):
            done = True
            reward = -5.0  # Penalidade por sair do campo
            return reward, done

        # 2. Atingiu o alvo (GRANDE RECOMPENSA)
        if (position_distance < self.position_tolerance and
                angular_distance < self.angular_tolerance):
            done = True
            # Recompensa baseada no tempo, o prêmio principal!
            reward = 500.0 * (1.0 - (self.steps / self.max_steps))
            return reward, done

        # 3. Timeout (Prêmio de Consolação por Proximidade)
        if self.steps >= self.max_steps:
            done = True

            # Recompensa pela proximidade da posição final (máximo de 10 pontos)
            reward_pos = 10 * np.exp(-1.5 * position_distance)

            # Recompensa pela proximidade do ângulo final (máximo de 5 pontos)
            reward_ang = 5 * np.exp(-1.0 * angular_distance)

            reward = reward_pos + reward_ang
            return reward, done

        # --- Shaping Reward (Se o episódio não terminou) ---

        # Recompensa por se APROXIMAR do alvo (positiva se melhora, negativa se piora)
        # Apenas calculamos se já temos uma distância anterior para comparar
        if self.previous_distance is not None:
            # Usamos um fator pequeno (ex: 5.0) para que a recompensa seja sutil
            progress_reward = 5.0 * (self.previous_distance - position_distance)
            reward += progress_reward

        # Atualiza a distância anterior para o próximo passo
        self.previous_distance = position_distance

        # Uma pequena penalidade constante por passo para incentivar a rapidez
        reward -= 0.01

        return reward, done
    def _is_out_of_bounds(self, robot):
        """Verifica se o robô saiu dos limites do campo"""
        x_limit = self.field.length / 2
        y_limit = self.field.width / 2
        return abs(robot.x) > x_limit or abs(robot.y) > y_limit

    def _get_initial_positions_frame(self) -> Frame:
        """Define as posições iniciais"""
        pos_frame = Frame()
        x_limit = self.field.length / 2 - 0.1
        y_limit = self.field.width / 2 - 0.1

        initial_pose = Pose(
            x=np.random.uniform(-x_limit, x_limit),
            y=np.random.uniform(-y_limit, y_limit),
            theta=np.random.uniform(-np.pi, np.pi)
        )

        min_distance = 0.5
        while True:
            target_pose = Pose(
                x=np.random.uniform(-x_limit, x_limit),
                y=np.random.uniform(-y_limit, y_limit),
                theta=np.random.uniform(-np.pi, np.pi)
            )
            if initial_pose.distance_to(target_pose) >= min_distance:
                break

        self.target_pose = target_pose
        self.initial_distance = initial_pose.distance_to(target_pose)
        self.initial_angular_distance = initial_pose.angular_distance_to(target_pose)

        pos_frame.robots_blue = {
            0: Robot(
                yellow=False, id=0,
                x=initial_pose.x, y=initial_pose.y,
                theta=np.rad2deg(initial_pose.theta),
            )
        }
        pos_frame.robots_yellow = {}
        pos_frame.ball = Ball(x=x_limit, y=y_limit)

        return pos_frame

    def _draw_target_pose(self, screen):
        """Desenha a pose alvo na tela"""
        if not hasattr(self, 'field_renderer') or self.target_pose is None:
            return

        def pos_transform(x, y):
            return (
                int(x * self.field_renderer.scale + self.field_renderer.center_x),
                int(y * self.field_renderer.scale + self.field_renderer.center_y)
            )

        target_x, target_y = pos_transform(self.target_pose.x, self.target_pose.y)

        radius = int(self.position_tolerance * self.field_renderer.scale)
        pygame.draw.circle(screen, COLORS["GREEN"], (target_x, target_y), radius, 2)

        arrow_length = 30
        arrow_end_x = target_x + int(arrow_length * np.cos(self.target_pose.theta))
        arrow_end_y = target_y + int(arrow_length * np.sin(self.target_pose.theta))
        pygame.draw.line(screen, COLORS["GREEN"], (target_x, target_y), (arrow_end_x, arrow_end_y), 3)

        angle1 = self.target_pose.theta + 2.5
        angle2 = self.target_pose.theta - 2.5
        arrow_size = 10
        point1 = (arrow_end_x - int(arrow_size * np.cos(angle1)), arrow_end_y - int(arrow_size * np.sin(angle1)))
        point2 = (arrow_end_x - int(arrow_size * np.cos(angle2)), arrow_end_y - int(arrow_size * np.sin(angle2)))
        pygame.draw.polygon(screen, COLORS["GREEN"], [(arrow_end_x, arrow_end_y), point1, point2])

    def _render(self):
        """Renderiza o ambiente incluindo a pose alvo"""
        super()._render()
        if hasattr(self, 'window_surface'):
            self._draw_target_pose(self.window_surface)
