import numpy as np
import gymnasium as gym
from rsoccer_gym.Entities import Ball, Frame, Robot
from playgrounds.el_ssl import CustomSSLEnv

class ELSSLReachBallEnv(CustomSSLEnv):
    """O robô precisa alcançar a bola no campo customizado EL

    Description:
        O robô deve se mover até a bola da maneira mais eficiente possível no campo EL (2.00m x 1.50m)

    Observation:
        Type: Box(11,)
        Num     Observation                                 
        0       Bola - Posição X                           
        1       Bola - Posição Y                           
        2       Bola - Velocidade X                        
        3       Bola - Velocidade Y                        
        4       Robô - Posição X                           
        5       Robô - Posição Y                           
        6       Robô - Velocidade Linear X                 
        7       Robô - Velocidade Linear Y                 
        8       Robô - sin(theta)                          
        9       Robô - cos(theta)                          
        10      Robô - Velocidade Angular                  

    Actions:
        Type: Box(3,)
        Num     Action
        0       Velocidade Linear X  (-1 a 1)
        1       Velocidade Linear Y  (-1 a 1)
        2       Velocidade Angular   (-1 a 1)

    Reward:
        - Recompensa positiva por se aproximar da bola (+10 * delta_distância)
        - Recompensa por alcançar a bola (+100)
        - Penalidade por tempo (-0.1 por step)
    """
    
    def __init__(self, render_mode=None, max_steps=1200):
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.distance_to_ball_start = None
        
        print("\n=== Ambiente EL Reach Ball ===")
        print(f"Campo: {self.field.length}m x {self.field.width}m")
        print(f"Distância mínima inicial robô-bola: 0.5m")
        print("=============================\n")

    def _calculate_reward_and_done(self):
        """Calcula a recompensa e condição de término"""
        reward = 0
        done = False
        
        robot = self.frame.robots_blue[0]
        ball = self.frame.ball

        # Calcula a distância atual até a bola
        current_distance = np.linalg.norm(
            np.array([robot.x - ball.x, robot.y - ball.y])
        )

        # Recompensa por se aproximar da bola
        if self.distance_to_ball_start is not None:
            reward += (self.distance_to_ball_start - current_distance) * 10
            self.distance_to_ball_start = current_distance

        # Recompensa por alcançar a bola (usando o raio do robô EL)
        robot_radius = 0.09  # Raio do robô EL
        ball_radius = 0.0215  # Raio da bola
        if current_distance < robot_radius + ball_radius + 0.1:
            reward += 100
            done = True

        # Penalidade por tempo
        reward -= 0.1

        # Término por tempo máximo
        if self.steps >= self.max_steps:
            done = True

        return reward, done

    def _get_initial_positions_frame(self) -> Frame:
        """Define as posições iniciais dos robôs e da bola"""
        pos_frame = Frame()

        # Calcular limites baseados no tamanho do campo EL
        x_limit = (self.field.length / 2) - 0.2  # 0.2m de margem
        y_limit = (self.field.width / 2) - 0.2   # 0.2m de margem

        # Posiciona a bola aleatoriamente no campo
        pos_frame.ball = Ball(
            x=np.random.uniform(-x_limit, x_limit),
            y=np.random.uniform(-y_limit, y_limit)
        )

        # Posiciona o robô aleatoriamente, mas a uma distância mínima da bola
        while True:
            robot_x = np.random.uniform(-x_limit, x_limit)
            robot_y = np.random.uniform(-y_limit, y_limit)

            distance_robot_ball = np.linalg.norm(
                np.array([
                    robot_x - pos_frame.ball.x,
                    robot_y - pos_frame.ball.y
                ])
            )

            if distance_robot_ball > 0.5:  # Distância mínima inicial
                pos_frame.robots_blue = {
                    0: Robot(
                        yellow=False,
                        id=0,
                        x=robot_x,
                        y=robot_y,
                        theta=np.random.uniform(-180, 180),  # Orientação aleatória
                    )
                }
                self.distance_to_ball_start = distance_robot_ball
                break

        print(f"Posições iniciais:")
        print(f"Bola: ({pos_frame.ball.x:.2f}, {pos_frame.ball.y:.2f})")
        print(f"Robô: ({robot_x:.2f}, {robot_y:.2f})")
        print(f"Distância inicial: {self.distance_to_ball_start:.2f}m\n")

        return pos_frame

    def reset(self, seed=None, options=None):
        """Reseta o ambiente e retorna a observação inicial"""
        self.distance_to_ball_start = None
        return super().reset(seed=seed, options=options)