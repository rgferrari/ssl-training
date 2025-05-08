import numpy as np
import gymnasium as gym
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Entities import Ball, Frame, Robot, Field

class CustomSSLEnv(SSLBaseEnv):
    """Custom SSL environment with modified robot parameters"""
    
    def __init__(self, render_mode=None, max_steps=1200):
        # Chamar o construtor pai com field_type=3 (EL field)
        super().__init__(
            field_type=3,  # EL field type
            n_robots_blue=1,
            n_robots_yellow=0,
            time_step=0.025,
            render_mode=render_mode,
        )

        # Configurar o tipo de robô para EL
        # self.rsim.set_robot_type(1)  # 1 = EL_ROBOT

        # Se precisar ajustar algum parâmetro específico
        # robot_params = {
        #     'wheel0_angle': 30.0,
        #     'wheel1_angle': 150.0,
        #     'wheel2_angle': 225.0,
        #     'wheel3_angle': 315.0,
        #     'radius': 0.09,
        #     'wheel_radius': 0.03,
        #     'motor_max_rpm': 2000
        # }
        # self.rsim.set_robot_params(robot_params)

        # Resto do código permanece o mesmo...

        n_actions = 3
        self.action_space = gym.spaces.Box(
            low=-1, 
            high=1, 
            shape=(n_actions,),
            dtype=np.float32
        )
        
        n_obs = 11
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        self.max_steps = max_steps

        # # Prints para debug
        # print("\n=== Campo EL ===")
        # print(f"Comprimento: {self.field.length} metros")
        # print(f"Largura: {self.field.width} metros")
        # print("Configuração do Robô EL:")
        # print(f"Ângulos das rodas: {robot_params['wheel0_angle']}°, {robot_params['wheel1_angle']}°, {robot_params['wheel2_angle']}°, {robot_params['wheel3_angle']}°")
        # print(f"RPM máximo: {robot_params['motor_max_rpm']}")
        # print("=============\n")

    def _get_commands(self, action):
        """Converte ações em comandos para o robô"""
        return [Robot(
            yellow=False,
            id=0,
            v_x=action[0],
            v_y=action[1],
            v_theta=action[2]
        )]

    def _frame_to_observations(self):
        """Converte o estado do ambiente em observações"""
        ball, robot = self.frame.ball, self.frame.robots_blue[0]
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
            self.norm_w(robot.v_theta)
        ], dtype=np.float32)

    def _calculate_reward_and_done(self):
        """Calcula a recompensa e condição de término"""
        reward = 0
        done = False

        if self.steps >= self.max_steps:
            done = True
            reward -= self.steps * 0.01

        return reward, done

    def _get_initial_positions_frame(self) -> Frame:
        """Define as posições iniciais dos robôs e da bola"""
        pos_frame = Frame()

        # Calcular limites baseados no tamanho real do campo
        x_limit = self.field.length / 2 - 0.2  # 0.2m de margem
        y_limit = self.field.width / 2 - 0.2   # 0.2m de margem

        print(f"\nPosicionando objetos dentro dos limites: x=±{x_limit:.2f}m, y=±{y_limit:.2f}m")

        # Posiciona a bola aleatoriamente no campo
        pos_frame.ball = Ball(
            x=np.random.uniform(-x_limit, x_limit),
            y=np.random.uniform(-y_limit, y_limit)
        )
        print(f"Bola posicionada em: ({pos_frame.ball.x:.2f}, {pos_frame.ball.y:.2f})")

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

            if distance_robot_ball > 0.2:
                pos_frame.robots_blue = {
                    0: Robot(
                        yellow=False,
                        id=0,
                        x=robot_x,
                        y=robot_y,
                        theta=0,
                    )
                }
                print(f"Robô posicionado em: ({robot_x:.2f}, {robot_y:.2f})")
                break

        return pos_frame