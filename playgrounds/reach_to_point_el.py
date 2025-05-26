import pygame
import numpy as np
import gymnasium as gym
from dataclasses import dataclass

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render.utils import COLORS

# Importando suas classes personalizadas
from playgrounds.Env.SSLELSim import SSL_EL_Env


@dataclass
class Position:
    x: float
    y: float


class SSLELReachToPoint(SSL_EL_Env):
    """
        Environment where the robot must reach a random target point into the EL Field
    """
    def __init__(self, render_mode=None, max_steps=1200):
        super().__init__(render_mode=render_mode, n_robots_blue=1, n_robots_yellow=0)
        
        self.max_steps = max_steps
        self.target_position = Position(x=0.0, y=0.0)
        
        n_actions = 3
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(n_actions,), dtype=np.float32
        )
        
        n_obs = 48
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        self.max_v = 5
        self.max_w = 10

        # Métricas para logging
        self.episodes_metrics = {}
        self.path_length = 0.0
        self.velocities = np.zeros(self.max_steps, dtype=np.float32)  # Pré-alocar array
        self.velocity_count = 0
        self.accelerations = np.zeros(self.max_steps-1, dtype=np.float32)  # Pré-alocar array
        self.acceleration_count = 0
        self.positions = np.zeros((self.max_steps, 2), dtype=np.float32)  # Para armazenar posições (x,y)
        self.position_count = 0
        self.direction_changes = 0
        self.last_velocity_direction = None

        # Parâmetros de recompensa
        self.reward_target_reached = 10
        self.penalty_out_of_bounds = -20
        self.reward_time_factor = 5.0
        self.max_steps_per_meter = 40


        



    def _get_initial_positions_frame(self) -> Frame:
        """
        Initializes the positions of all entities for the episode.
        For curriculum learning, always allocate space for 3 blue and 3 yellow robots.
        Only the agent (blue[0]) will be present at the start; others will be absent (dummy values).
        """
        pos_frame = Frame()
        x_limit = self.field.length / 2 - 0.2
        y_limit = self.field.width / 2 - 0.2



        # Randomly sample the target position within the field limits
        self.target_position = Position(
            x=np.random.uniform(-x_limit, x_limit),
            y=np.random.uniform(-y_limit, y_limit),
        )

        # Place the agent (blue[0]) at a random position
        pos_frame.robots_blue = {}
        robot_x = np.random.uniform(-x_limit, x_limit)
        robot_y = np.random.uniform(-y_limit, y_limit)
        pos_frame.robots_blue[0] = Robot(
            yellow=False,
            id=0,
            x=robot_x,
            y=robot_y,
            theta=np.random.uniform(-180, 180),
        )
        
        # Resetar métricas para o novo episódio
        self.path_length = 0.0
        self.velocity_count = 0
        self.acceleration_count = 0
        self.position_count = 0
        self.direction_changes = 0
        self.last_velocity_direction = None
        self.steps_to_target = 0
        
        # Calcular a distância inicial entre o robô e o alvo
        self.initial_distance = np.linalg.norm([robot_x - self.target_position.x, robot_y - self.target_position.y])
        
        # Armazenar posição inicial
        self.positions[0] = [robot_x, robot_y]
        self.position_count = 1

        # Other blue robots are not present in this phase
        # They will be handled as dummy in the observation

        # Ball is not used, but kept for compatibility (set at origin)
        pos_frame.ball = Ball(x=0, y=0)

        # No yellow robots present at this phase
        pos_frame.robots_yellow = {}

        return pos_frame

    def _frame_to_observations(self):
        """
        Converts the current environment state to the observation vector.
        For curriculum learning, always includes all 3 blue and 3 yellow robots in a fixed order.
        Absent robots are filled with dummy values.
        Observation vector:
        [ball(4), blue robots(3x7), yellow robots(3x7), target(2)] = 48 elements
        """
        obs = []

        # Ball: [x, y, v_x, v_y] (normalized)
        ball = self.frame.ball
        obs.extend([
            self.norm_pos(ball.x), self.norm_pos(ball.y),
            self.norm_v(ball.v_x), self.norm_v(ball.v_y)
        ])

        # Blue robots (agent and teammates): always 3 slots
        for i in range(3):
            if i in self.frame.robots_blue:
                robot = self.frame.robots_blue[i]
                obs.extend([
                    self.norm_pos(robot.x), self.norm_pos(robot.y),
                    self.norm_v(robot.v_x), self.norm_v(robot.v_y),
                    np.sin(np.deg2rad(robot.theta)), np.cos(np.deg2rad(robot.theta)),
                    self.norm_w(robot.v_theta)
                ])
            else:
                # Dummy values for absent blue robots
                obs.extend([0, 0, 0, 0, 0, 1, 0])

        # Yellow robots (opponents): always 3 slots
        for i in range(3):
            if i in self.frame.robots_yellow:
                robot = self.frame.robots_yellow[i]
                obs.extend([
                    self.norm_pos(robot.x), self.norm_pos(robot.y),
                    self.norm_v(robot.v_x), self.norm_v(robot.v_y),
                    np.sin(np.deg2rad(robot.theta)), np.cos(np.deg2rad(robot.theta)),
                    self.norm_w(robot.v_theta)
                ])
            else:
                # Dummy values for absent yellow robots
                obs.extend([0, 0, 0, 0, 0, 1, 0])

        # Target position (normalized)
        obs.extend([
            self.norm_pos(self.target_position.x),
            self.norm_pos(self.target_position.y)
        ])

        return np.array(obs, dtype=np.float32)


    def _get_commands(self, action):
        """
        Converts the action vector into robot commands for the agent (blue[0]).
        The action vector is expected to be [v_x, v_y, v_theta], all normalized to [-1, 1].
        Returns a list of Robot commands (only the agent is controlled here).
        """
        angle = self.frame.robots_blue[0].theta
        v_x, v_y, v_theta = self.convert_actions(action, np.deg2rad(angle))
        return [Robot(yellow=False, id=0, v_x=v_x, v_y=v_y, v_theta=v_theta)]
        
    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        
        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle), \
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x, v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def _calculate_reward_and_done(self):
        """
        Calculates the reward and termination condition for the current step.
        Includes calculation of various metrics for logging and visualization.
        """
        robot = self.frame.robots_blue[0]
        target = self.target_position

        # Calcular distância até o alvo
        dist_to_target = np.linalg.norm([robot.x - target.x, robot.y - target.y])

        # Recompensa base é negativa da distância (quanto menor a distância, melhor)
        reward = -dist_to_target

        done = False
        self.steps_to_target += 1

        # Calcular métricas adicionais
        current_position = np.array([robot.x, robot.y], dtype=np.float32)
        
        if self.position_count > 0:
            # Calcular distância do último passo usando NumPy
            last_position = self.positions[self.position_count-1]
            step_distance = np.linalg.norm(current_position - last_position)
            self.path_length += step_distance
        
        # Armazenar posição atual
        if self.position_count < self.max_steps:
            self.positions[self.position_count] = current_position
            self.position_count += 1
        
        # Calcular e armazenar velocidade
        velocity = np.linalg.norm([robot.v_x, robot.v_y])
        if self.velocity_count < self.max_steps:
            self.velocities[self.velocity_count] = velocity
            self.velocity_count += 1
        
        # Calcular e armazenar aceleração
        if self.velocity_count > 1:
            acceleration = abs(self.velocities[self.velocity_count-1] - self.velocities[self.velocity_count-2]) / self.time_step
            if self.acceleration_count < self.max_steps-1:
                self.accelerations[self.acceleration_count] = acceleration
                self.acceleration_count += 1
        
        # Detectar mudanças de direção usando NumPy
        if velocity > 0.1:  # Ignorar velocidades muito baixas
            current_direction = np.arctan2(robot.v_y, robot.v_x)
            if self.last_velocity_direction is not None:
                angle_diff = np.abs(np.angle(np.exp(1j * (current_direction - self.last_velocity_direction))))
                if angle_diff > np.pi/4:  # Mudança de mais de 45 graus
                    self.direction_changes += 1
            self.last_velocity_direction = current_direction

        # Verificar se o robô saiu do campo
        field_length_half = self.field.length / 2
        field_width_half = self.field.width / 2
        out_of_bounds = (
            abs(robot.x) > field_length_half or 
            abs(robot.y) > field_width_half
        )
        
        # Atualizar métricas do episódio
        success = dist_to_target < 0.1
        timeout = self.steps >= self.max_steps
        
        # Usar apenas os dados válidos para cálculos de média
        valid_velocities = self.velocities[:self.velocity_count]
        valid_accelerations = self.accelerations[:self.acceleration_count]
        
        avg_velocity = np.mean(valid_velocities) if self.velocity_count > 0 else 0
        avg_acceleration = np.mean(valid_accelerations) if self.acceleration_count > 0 else 0
        path_efficiency = self.initial_distance / self.path_length if self.path_length > 0 else 0
        
        # Verificar condições de término e calcular recompensas
        if out_of_bounds:
            # Penalidade por sair do campo
            reward += self.penalty_out_of_bounds
            done = True
        
        # Verificar se o robô alcançou o alvo
        if success:
            # Recompensa base por alcançar o alvo
            reward += self.reward_target_reached
            
            # Recompensa adicional baseada no tempo normalizado pela distância inicial
            # Calculamos o número máximo de passos esperados com base na distância inicial
            expected_max_steps = self.initial_distance * self.max_steps_per_meter
            
            # Quanto mais rápido o robô chegar (menos passos), maior a recompensa
            # Normalizamos pela distância inicial para ser justo independente da distância
            if expected_max_steps > 0:
                time_efficiency = max(0, 1 - (self.steps_to_target / expected_max_steps))
                time_reward = self.reward_time_factor * time_efficiency
                reward += time_reward
            
            done = True
        
        # Verificar se atingiu o número máximo de passos
        if timeout:
            done = True
        
        # Armazenar métricas completas para logging
        self.episodes_metrics = {
            'success': success,
            'out_of_bounds': out_of_bounds,
            'timeout': timeout,
            'initial_distance': self.initial_distance,
            'final_distance': dist_to_target,
            'steps_to_target': self.steps_to_target,
            'steps_per_meter': self.steps_to_target / self.initial_distance if self.initial_distance > 0 else 0,
            'path_length': self.path_length,
            'path_efficiency': path_efficiency,
            'avg_velocity': avg_velocity,
            'avg_acceleration': avg_acceleration,
            'direction_changes': self.direction_changes,
            'episode_reward': reward
        }
        
        # Criar info para retornar ao ambiente
        info = {}
        if done:
            info['episode_metrics'] = self.episodes_metrics.copy()
            print(f"[Environment] Episode metrics being returned: {info['episode_metrics']}")  # Debug print

        return reward, done

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            # Certifique-se de que 'episode_metrics' é um dicionário serializável!
            info['episode_metrics'] = self.episodes_metrics.copy()
            print(f"[Environment] Episode metrics being returned: {info['episode_metrics']}")
        return obs, reward, terminated, truncated, info

    def _draw_target(self, target_pos, screen):
        """
        Draws the target area as a square on the field during rendering.
        The square is sized so that the robot (radius 0.09m) can fit entirely inside.
        """
        # Robot radius in meters
        robot_radius = self.field.rbt_radius  # Should be 0.09m
        # Size of the square (side = 2 * robot_radius)
        square_side_m = 2 * robot_radius

        # Convert field coordinates to screen coordinates
        center_x, center_y = target_pos
        scale = self.field_renderer.scale

        # Calculate top-left corner of the square in screen coordinates
        top_left_x = int(center_x * scale + self.field_renderer.center_x - (square_side_m * scale) / 2)
        top_left_y = int(center_y * scale + self.field_renderer.center_y - (square_side_m * scale) / 2)
        square_side_px = int(square_side_m * scale)

        # Draw filled red square
        pygame.draw.rect(screen, COLORS["RED"], (top_left_x, top_left_y, square_side_px, square_side_px))
        # Draw black border
        pygame.draw.rect(screen, COLORS["BLACK"], (top_left_x, top_left_y, square_side_px, square_side_px), 2)
        
    def _render(self):
        super()._render()
        # Draw the target area as a square
        self._draw_target([self.target_position.x, self.target_position.y], self.window_surface)

        # Print robot velocity if episode is done (or always, if you prefer)
        robot = self.frame.robots_blue[0]
        v_x = robot.v_x
        v_y = robot.v_y
        v_theta = robot.v_theta
        speed = np.sqrt(v_x**2 + v_y**2)
        # print(f"[RENDER] Robot velocity: v_x={v_x:.3f} m/s, v_y={v_y:.3f} m/s, linear speed={speed:.3f} m/s, v_theta={v_theta:.3f} rad/s")