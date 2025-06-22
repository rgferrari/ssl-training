import pygame
import numpy as np
import gymnasium as gym
from dataclasses import dataclass

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render.utils import COLORS

# Importando sua classe base
from playgrounds.Env.SSLELSim import SSL_EL_Env


class SSLELGoalkeeperEnv(SSL_EL_Env):
    """Ambiente especializado para treinar um goleiro SSL

    Description:
        O robô goleiro deve defender chutes direcionados ao gol.
        A bola é lançada de posições aleatórias em direção ao gol.

    Observation:
        Type: Box(12,)
        Normalized Bounds to [-1.2, 1.2]
        Num      Observation normalized
        0->3     Ball [X, Y, V_x, V_y]
        4->10    Robot [X, Y, V_x, V_y, sin_theta, cos_theta, v_theta]
        11       Distance to goal line

    Actions:
        Type: Box(3,)
        Value Range: [-1, 1]
        Num     Action
        0       V_X
        1       V_y
        2       V_theta

    Reward:
        - Recompensa por estar entre a bola e o gol
        - Recompensa por diminuir a distância até a trajetória da bola
        - Penalidade pesada se tomar gol
        - Bônus por defesa bem-sucedida

    Starting State:
        - Goleiro começa próximo ao gol
        - Bola é lançada de posição aleatória em direção ao gol

    Episode Termination:
        - Quando a bola cruza a linha do gol (gol ou defesa)
        - Quando a bola sai do campo
        - Após 600 timesteps (15 segundos)
    """

    def __init__(self, render_mode=None, max_steps=600):
        # Chama o construtor da classe pai
        super().__init__(render_mode=render_mode, n_robots_blue=1, n_robots_yellow=0)

        # Configura os espaços de ação e observação
        n_actions = 3
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(n_actions,), dtype=np.float32
        )

        n_obs = 12
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        self.max_steps = max_steps
        self.goal_width = 0.18  # 18cm - largura do gol SSL
        self.goal_depth = 0.18  # profundidade do gol
        self.initial_ball_distance = None
        self.ball_initial_velocity = None

    def _get_commands(self, action):
        """Converte ações em comandos para o robô"""
        return [
            Robot(yellow=False, id=0, v_x=action[0], v_y=action[1], v_theta=action[2])
        ]

    def _frame_to_observations(self):
        """Converte o estado do ambiente em observações"""
        ball, robot = self.frame.ball, self.frame.robots_blue[0]

        # Calcular distância do robô até a linha do gol
        goal_x = -self.field.length / 2
        distance_to_goal_line = abs(robot.x - goal_x)

        return np.array([
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
            self.norm_pos(distance_to_goal_line),
        ], dtype=np.float32)

    def _calculate_reward_and_done(self):
        """Calcula a recompensa e condição de término"""
        ball, robot = self.frame.ball, self.frame.robots_blue[0]

        reward = 0
        done = False

        # Posição do gol
        goal_x = -self.field.length / 2
        goal_y_min = -self.goal_width / 2
        goal_y_max = self.goal_width / 2

        # Verificar se a bola cruzou a linha do gol
        if ball.x <= goal_x:
            done = True
            if goal_y_min <= ball.y <= goal_y_max:
                # Gol! Grande penalidade
                reward -= 100
            else:
                # Bola saiu pela linha de fundo mas não foi gol - defesa bem sucedida
                reward += 50

        # Verificar se a bola saiu do campo pelos lados
        if abs(ball.y) > self.field.width / 2:
            done = True
            reward += 30  # Defesa parcial - forçou a bola para fora

        # Verificar timeout
        if self.steps >= self.max_steps:
            done = True
            reward += 20  # Sobreviveu ao tempo todo

        if not done:
            # Recompensas contínuas durante o jogo

            # 1. Recompensa por estar entre a bola e o gol
            if robot.x < ball.x and robot.x > goal_x:
                # Verificar se está alinhado com a bola no eixo Y
                alignment_bonus = np.exp(-abs(robot.y - ball.y) / 0.1)
                reward += 0.5 * alignment_bonus

            # 2. Recompensa por estar próximo à trajetória da bola
            if ball.v_x < -0.01:  # Bola indo em direção ao gol
                # Prever onde a bola vai estar quando chegar na posição X do robô
                time_to_robot = (robot.x - ball.x) / ball.v_x if ball.v_x != 0 else 0
                predicted_y = ball.y + ball.v_y * time_to_robot

                # Recompensa por estar próximo à posição prevista
                distance_to_trajectory = abs(robot.y - predicted_y)
                reward += 0.3 * np.exp(-distance_to_trajectory / 0.2)

            # 3. Penalidade por estar muito longe do gol
            ideal_x = goal_x + 0.15  # 15cm na frente do gol
            distance_from_ideal = abs(robot.x - ideal_x)
            if distance_from_ideal > 0.3:  # Mais de 30cm de distância
                reward -= 0.1 * distance_from_ideal

            # 4. Penalidade por movimento desnecessário quando a bola está parada
            if abs(ball.v_x) < 0.01 and abs(ball.v_y) < 0.01:
                robot_speed = np.sqrt(robot.v_x**2 + robot.v_y**2)
                reward -= 0.05 * robot_speed

        return reward, done

    def _get_initial_positions_frame(self) -> Frame:
        """Define as posições iniciais dos robôs e da bola"""
        pos_frame = Frame()

        # Posição do gol
        goal_x = -self.field.length / 2

        # Goleiro começa próximo ao gol
        goalkeeper_x = goal_x + 0.15  # 15cm na frente do gol
        goalkeeper_y = np.random.uniform(-0.05, 0.05)  # Pequena variação no Y
        goalkeeper_theta = 0  # Olhando para frente

        pos_frame.robots_blue = {
            0: Robot(
                yellow=False,
                id=0,
                x=goalkeeper_x,
                y=goalkeeper_y,
                theta=goalkeeper_theta,
            )
        }

        # Bola começa em posição aleatória no campo adversário
        ball_x = np.random.uniform(0, self.field.length / 2 - 0.3)
        ball_y = np.random.uniform(-self.field.width / 3, self.field.width / 3)

        # Calcular velocidade da bola para mirar no gol (com alguma variação)
        # Adicionar ruído para tornar mais desafiador
        target_y = np.random.uniform(-self.goal_width / 2 * 1.5, self.goal_width / 2 * 1.5)
        direction_x = goal_x - ball_x
        direction_y = target_y - ball_y

        # Normalizar e aplicar velocidade
        distance = np.sqrt(direction_x**2 + direction_y**2)
        speed = np.random.uniform(0.8, 1.5)  # Velocidade entre 0.8 e 1.5 m/s

        ball_vx = speed * direction_x / distance
        ball_vy = speed * direction_y / distance

        pos_frame.ball = Ball(
            x=ball_x,
            y=ball_y,
            v_x=ball_vx,
            v_y=ball_vy
        )

        # Salvar velocidade inicial para análise
        self.ball_initial_velocity = np.sqrt(ball_vx**2 + ball_vy**2)
        self.initial_ball_distance = distance

        # Robôs amarelos vazios
        pos_frame.robots_yellow = {}

        return pos_frame

    def step(self, action):
        """Override do step para garantir que a bola se move"""
        # Chamar step do pai
        obs, reward, terminated, truncated, info = super().step(action)

        # Atualizar posição da bola manualmente se necessário
        # (caso o simulador pai não esteja fazendo isso)
        if hasattr(self, 'frame') and self.frame.ball:
            ball = self.frame.ball
            dt = 1.0 / 40.0  # 40 FPS típico

            # Atualizar posição baseada na velocidade
            ball.x += ball.v_x * dt
            ball.y += ball.v_y * dt

            # Aplicar fricção simples
            friction = 0.98
            ball.v_x *= friction
            ball.v_y *= friction

        return obs, reward, terminated, truncated, info

    def _draw_goal(self, screen):
        """Desenha o gol na tela"""
        if not hasattr(self, 'field_renderer'):
            return

        # Posição do gol em coordenadas da tela
        goal_x = -self.field.length / 2
        goal_y_min = -self.goal_width / 2
        goal_y_max = self.goal_width / 2

        # Converter para coordenadas da tela
        def pos_transform(x, y):
            return (
                int(x * self.field_renderer.scale + self.field_renderer.center_x),
                int(y * self.field_renderer.scale + self.field_renderer.center_y)
            )

        # Desenhar linha do gol
        p1 = pos_transform(goal_x, goal_y_min)
        p2 = pos_transform(goal_x, goal_y_max)
        pygame.draw.line(screen, COLORS["RED"], p1, p2, 3)

        # Desenhar área do gol (profundidade)
        back_x = goal_x - self.goal_depth
        p3 = pos_transform(back_x, goal_y_min)
        p4 = pos_transform(back_x, goal_y_max)

        # Linhas laterais do gol
        pygame.draw.line(screen, COLORS["RED"], p1, p3, 2)
        pygame.draw.line(screen, COLORS["RED"], p2, p4, 2)
        pygame.draw.line(screen, COLORS["RED"], p3, p4, 2)

    def _render(self):
        """Renderiza o ambiente incluindo o gol"""
        super()._render()

        if hasattr(self, 'window_surface'):
            self._draw_goal(self.window_surface)
