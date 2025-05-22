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


class SSLELReachBallEnv(SSL_EL_Env):
    """O robô SSL precisa alcançar a bola no campo EL personalizado

    Description:
        O robô deve se mover até a bola e depois até o alvo

    Observation:
        Type: Box(13,)
        Normalized Bounds to [-1.2, 1.2]
        Num      Observation normalized
        0->3     Ball [X, Y, V_x, V_y]
        4->10    Robot [X, Y, V_x, V_y, sin_theta, cos_theta, v_theta]
        11->12   Target [X, Y]

    Actions:
        Type: Box(3,)
        Value Range: [-1, 1]
        Num     Action
        0       V_X
        1       V_y
        2       V_theta

    Reward:
     - Recompensa por se aproximar da bola
     - Recompensa por tocar na bola
     - Recompensa por estar orientado na direção da bola/alvo

    Starting State:
        - Posições do robô e da bola são aleatórias

    Episode Termination:
        30 segundos (1200 passos) ou quando alcançar o alvo
    """

    def __init__(self, render_mode=None, max_steps=1200):
        # Chama o construtor da classe pai (SSL_EL_Env)
        super().__init__(render_mode=render_mode, n_robots_blue=1, n_robots_yellow=0)

        # Configura os espaços de ação e observação específicos para este ambiente
        n_actions = 3
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(n_actions,), dtype=np.float32
        )

        n_obs = 13  # Inclui posição do alvo
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        self.max_steps = max_steps
        self.target_position = Position(x=0.0, y=0.0)
        self.distance_robot_ball_start = None

    def _get_commands(self, action):
        """Converte ações em comandos para o robô"""
        return [
            Robot(yellow=False, id=0, v_x=action[0], v_y=action[1], v_theta=action[2])
        ]

    def _frame_to_observations(self):
        """Converte o estado do ambiente em observações"""
        ball, robot = self.frame.ball, self.frame.robots_blue[0]
        return np.array(
            [
                self.norm_pos(ball.x),
                self.norm_pos(ball.y),
                self.norm_v(ball.v_x),
                self.norm_v(ball.v_y),
                self.norm_pos(robot.x),
                self.norm_pos(robot.y),
                self.norm_v(robot.v_x),
                self.norm_v(robot.v_y),
                np.sin(np.deg2rad(robot.theta)),
                np.cos(np.deg2rad(robot.theta)),
                self.norm_w(robot.v_theta),
                self.norm_pos(self.target_position.x),
                self.norm_pos(self.target_position.y),
            ],
            dtype=np.float32,
        )

    def _calculate_reward_and_done(self):
        """Calcula a recompensa e condição de término"""
        ball, robot = self.frame.ball, self.frame.robots_blue[0]

        reward = 0
        done = False

        # Recompensa se tocou na bola (detectado pela velocidade da bola)
        if ball.v_x > 0.05 or ball.v_y > 0.05:
            reward += 10
            done = True

        if self.steps >= self.max_steps:
            done = True

        if done:
            # Penalidade pelo tempo gasto
            reward -= self.steps * 0.01

            # Recompensa se o robô está olhando para a bola
            if self._robot_is_facing(ball):
                reward += 5

            # Recompensa se o robô está olhando para o alvo
            if self._robot_is_facing(self.target_position):
                reward += 5

            # Calcula a distância atual entre o robô e a bola
            distance_robot_ball = np.linalg.norm(
                np.array([robot.x - ball.x, robot.y - ball.y])
            )

            # Recompensa se o robô se aproximou da bola
            reward += (self.distance_robot_ball_start - distance_robot_ball) * 5

        return reward, done

    def _get_initial_positions_frame(self) -> Frame:
        """Define as posições iniciais dos robôs e da bola"""
        pos_frame = Frame()

        # Calcular limites baseados no tamanho do campo
        x_limit = self.field.length / 2 - 0.2  # 0.2m de margem
        y_limit = self.field.width / 2 - 0.2  # 0.2m de margem

        # Posiciona a bola aleatoriamente no campo
        pos_frame.ball = Ball(
            x=np.random.uniform(-x_limit, x_limit),
            y=np.random.uniform(-y_limit, y_limit),
        )

        # Define uma posição alvo aleatória
        self.target_position = Position(
            x=np.random.uniform(-x_limit, x_limit),
            y=np.random.uniform(-y_limit, y_limit),
        )

        # Posiciona o robô aleatoriamente, mas a uma distância mínima da bola
        pos_frame.robots_blue = {}
        while True:
            robot_x = np.random.uniform(-x_limit, x_limit)
            robot_y = np.random.uniform(-y_limit, y_limit)

            distance_robot_ball = np.linalg.norm(
                np.array([robot_x - pos_frame.ball.x, robot_y - pos_frame.ball.y])
            )

            if distance_robot_ball > 0.2:
                pos_frame.robots_blue[0] = Robot(
                    yellow=False,
                    id=0,
                    x=robot_x,
                    y=robot_y,
                    theta=np.random.uniform(-180, 180),
                )
                self.distance_robot_ball_start = distance_robot_ball
                break

        # Inicializa robôs amarelos (vazio neste caso)
        pos_frame.robots_yellow = {}

        return pos_frame

    def _robot_is_facing(self, target):
        """Verifica se o robô está orientado na direção do alvo"""
        robot = self.frame.robots_blue[0]
        theta_rad = np.deg2rad(robot.theta)

        robot_to_target_direction = np.array([target.x - robot.x, target.y - robot.y])
        # Evita divisão por zero
        norm = np.linalg.norm(robot_to_target_direction)
        if norm < 1e-6:
            return False

        robot_to_target_direction /= norm
        robot_orientation = np.array([np.cos(theta_rad), np.sin(theta_rad)])
        dot_product = np.dot(robot_orientation, robot_to_target_direction)

        return (
            dot_product > 0.9
        )  # Consideramos que está olhando se o ângulo é menor que ~25 graus

    def _draw_target(self, target_pos, screen):
        """Desenha o alvo na tela"""
        target_radius = self.field.ball_radius * self.field_renderer.scale
        target_x, target_y = target_pos
        pygame.draw.circle(screen, COLORS["RED"], (target_x, target_y), target_radius)
        pygame.draw.circle(
            screen, COLORS["BLACK"], (target_x, target_y), target_radius, 1
        )

    def _render(self):
        """Renderiza o ambiente incluindo o alvo"""
        super()._render()

        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
                int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
            )

        target_x, target_y = pos_transform(
            self.target_position.x, self.target_position.y
        )
        self._draw_target([target_x, target_y], self.window_surface)
