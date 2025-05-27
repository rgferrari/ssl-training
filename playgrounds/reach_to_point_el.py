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
        super().__init__(render_mode=render_mode, n_robots_blue=3, n_robots_yellow=3)
        
        self.max_steps = max_steps
        self.target_position = Position(x=0.0, y=0.0)
        
        # Ações: v_x, v_y, v_theta, kick_v_x, dribbler
        n_actions = 5
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(n_actions,), dtype=np.float32
        )
        
        # Observações: bola(4) + robô controlado(8) + outros robôs azuis(2x2) + robôs amarelos(3x2) + alvo(3) = 23
        n_obs = 4 + 8 + 2 * (self.n_robots_blue - 1) + 2 * self.n_robots_yellow + 3
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        self.max_v = 5.0
        self.max_w = 10
        self.kick_speed_x = 5.0

        # Métricas para logging
        self.episodes_metrics = {}
        self.steps_to_target = 0
        self.initial_distance = 0.0
        
        # Parâmetros de recompensa
        self.reward_target_reached = 100.0
        self.penalty_out_of_bounds = -20.0
        self.max_steps_per_meter = 12.0
        self.penalty_per_step = -0.1
        self.penalty_timeout = -30.0

    def _get_initial_positions_frame(self) -> Frame:
        """
        Initializes the positions of all entities for the episode.
        Always allocate space for 3 blue and 3 yellow robots.
        Only the agent (blue[0]) will be actively controlled.
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
        
        # Adicionar robôs azuis extras (2 e 3) em posições aleatórias
        for i in range(1, self.n_robots_blue):
            pos_frame.robots_blue[i] = Robot(
                yellow=False,
                id=i,
                x=np.random.uniform(-x_limit, x_limit),
                y=np.random.uniform(-y_limit, y_limit),
                theta=np.random.uniform(-180, 180),
            )
        
        # Adicionar robôs amarelos em posições aleatórias
        pos_frame.robots_yellow = {}
        for i in range(self.n_robots_yellow):
            pos_frame.robots_yellow[i] = Robot(
                yellow=True,
                id=i,
                x=np.random.uniform(-x_limit, x_limit),
                y=np.random.uniform(-y_limit, y_limit),
                theta=np.random.uniform(-180, 180),
            )
        
        # Resetar métricas para o novo episódio
        self.steps_to_target = 0
        
        # Calcular a distância inicial entre o robô e o alvo
        self.initial_distance = np.linalg.norm([robot_x - self.target_position.x, robot_y - self.target_position.y])
        
        # Ball is not used actively but kept for compatibility (set at origin)
        pos_frame.ball = Ball(x=0, y=0)

        return pos_frame

    def _frame_to_observations(self):
        """
        Converts the current environment state to the observation vector.
        Observation vector:
        [ball(4), blue[0](8), blue[1:3](2x2), yellow robots(3x2), target(3)] = 23 elements
        """
        obs = []

        # Ball: [x, y, v_x, v_y] (normalized)
        ball = self.frame.ball
        obs.extend([
            self.norm_pos(ball.x), self.norm_pos(ball.y),
            self.norm_v(ball.v_x), self.norm_v(ball.v_y)
        ])

        # Blue robot 0 (controlled agent): 8 valores
        if 0 in self.frame.robots_blue:
            robot = self.frame.robots_blue[0]
            obs.extend([
                self.norm_pos(robot.x), self.norm_pos(robot.y),
                np.sin(np.deg2rad(robot.theta)), np.cos(np.deg2rad(robot.theta)),
                self.norm_v(robot.v_x), self.norm_v(robot.v_y),
                self.norm_w(robot.v_theta),
                1 if robot.infrared else 0
            ])
        else:
            # Dummy values for absent blue robot 0
            obs.extend([0, 0, 0, 1, 0, 0, 0, 0])

        # Other blue robots (teammates): only x, y
        for i in range(1, self.n_robots_blue):
            if i in self.frame.robots_blue:
                robot = self.frame.robots_blue[i]
                obs.extend([
                    self.norm_pos(robot.x), self.norm_pos(robot.y)
                ])
            else:
                # Dummy values for absent blue robots
                obs.extend([0, 0])

        # Yellow robots (opponents): only x, y
        for i in range(self.n_robots_yellow):
            if i in self.frame.robots_yellow:
                robot = self.frame.robots_yellow[i]
                obs.extend([
                    self.norm_pos(robot.x), self.norm_pos(robot.y)
                ])
            else:
                # Dummy values for absent yellow robots
                obs.extend([0, 0])

        # Target: [x, y, theta] (normalized)
        target = self.target_position
        # Vamos usar um ângulo fixo para o alvo (0 graus)
        target_theta = 0.0
        obs.extend([
            self.norm_pos(target.x), 
            self.norm_pos(target.y),
            np.deg2rad(target_theta)  # theta em radianos
        ])

        return np.array(obs, dtype=np.float32)

    def _get_commands(self, action):
        """
        Converts the action vector into robot commands for the agent (blue[0]).
        The action vector is expected to be [v_x, v_y, v_theta, kick_v_x, dribbler], all normalized to [-1, 1].
        Returns a list of Robot commands (only the agent is controlled here).
        Neste primeiro momento, kick_v_x e dribbler são sempre 0, independentemente da ação.
        """
        angle = self.frame.robots_blue[0].theta
        v_x, v_y, v_theta = self.convert_actions(action[:3], np.deg2rad(angle))
        
        return [Robot(
            yellow=False, 
            id=0, 
            v_x=v_x, 
            v_y=v_y, 
            v_theta=v_theta,
            kick_v_x=0.0,  # Sempre 0, independentemente da ação
            dribbler=False  # Sempre False, independentemente da ação
        )]
        
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
        Simplified reward: reach target + time efficiency penalty
        """
        robot = self.frame.robots_blue[0]
        target = self.target_position

        # Calcular distância até o alvo
        dist_to_target = np.linalg.norm([robot.x - target.x, robot.y - target.y])

        # Recompensa base é negativa da distância (quanto menor a distância, melhor)
        reward = -dist_to_target

        done = False
        self.steps_to_target += 1
        
        # Verificar se o robô saiu do campo
        field_length_half = self.field.length / 2
        field_width_half = self.field.width / 2
        out_of_bounds = (
            abs(robot.x) > field_length_half or 
            abs(robot.y) > field_width_half
        )
        
        # Verificar condições de término e calcular recompensas
        if out_of_bounds:
            # Penalidade por sair do campo
            reward += self.penalty_out_of_bounds
            done = True
        
        # Verificar se o robô alcançou o alvo
        success = dist_to_target < 0.09
        if success:
            # Recompensa base por alcançar o alvo
            reward += self.reward_target_reached
            
            # Recompensa adicional baseada no tempo normalizado pela distância inicial
            # Calculamos o número máximo de passos esperados com base na distância inicial
            expected_max_steps = self.initial_distance * self.max_steps_per_meter
            
            # Quanto mais rápido o robô chegar (menos passos), maior a recompensa
            if expected_max_steps > 0:
                time_efficiency = max(0, 1 - (self.steps_to_target / expected_max_steps))
                time_reward = self.reward_target_reached * time_efficiency
                reward += time_reward
            
            done = True
        
        # Verificar se atingiu o número máximo de passos
        if self.steps_to_target >= self.max_steps:
            reward += self.penalty_timeout  # Penalidade adicional por timeout
            done = True
        
        # Penalidade por cada passo (incentiva o agente a ser mais rápido)
        reward += self.penalty_per_step
        
        # Armazenar métricas para logging
        self.episodes_metrics = {
            'success': success,
            'out_of_bounds': out_of_bounds,
            'timeout': self.steps_to_target >= self.max_steps,
            'initial_distance': self.initial_distance,
            'final_distance': dist_to_target,
            'steps_to_target': self.steps_to_target,
            'steps_per_meter': self.steps_to_target / self.initial_distance if self.initial_distance > 0 else 0,
            'episode_reward': reward
        }
        
        # Criar info para retornar ao ambiente
        info = {}
        if done:
            info['episode_metrics'] = self.episodes_metrics.copy()
        
        return reward, done

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            info['episode_metrics'] = self.episodes_metrics.copy()
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