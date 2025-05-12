import numpy as np
from dataclasses import replace
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Simulators.rsim import RSimSSL
import robosim
import gymnasium as gym
from playgrounds.Render.SSLELRenderField import SSLELRenderField as SSLRenderField
from rsoccer_gym.Entities import Robot
from rsoccer_gym.Entities import Frame, Ball, Robot


# Primeiro criamos uma classe personalizada que herda de RSimSSL
class SSLELSim(RSimSSL):
    def _init_simulator(
        self,
        field_type,
        n_robots_blue,
        n_robots_yellow,
        ball_pos,
        blue_robots_pos,
        yellow_robots_pos,
        time_step_ms,
    ):
        # Sempre usamos o tipo 2 como base
        simulator = robosim.SSL(
            2,
            n_robots_blue,
            n_robots_yellow,
            time_step_ms,
            ball_pos,
            blue_robots_pos,
            yellow_robots_pos,
        )

        # Não podemos modificar o simulador diretamente, mas podemos modificar o Field depois
        return simulator


# Agora criamos o ambiente personalizado
class SSL_EL_Env(SSLBaseEnv):
    def __init__(self, render_mode=None, n_robots_blue=1, n_robots_yellow=0):
        # Inicializa variáveis básicas
        self.render_mode = render_mode
        self.time_step = 0.025
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow

        # Cria o simulador personalizado
        self.rsim = SSLELSim(
            field_type=2,  # Usamos o tipo 2 como base
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step_ms=int(self.time_step * 1000),
        )

        # Obtém o campo original
        self.field_type = 2
        self.field = self.rsim.get_field_params()

        # Substitui o campo com os parâmetros personalizados
        self.field = replace(
            self.field,
            length=4.5,
            width=3.0,
            penalty_length=0.5,
            penalty_width=1.350,
            goal_width=0.8,
            goal_depth=0.18,
            ball_radius=0.0215,
            rbt_distance_center_kicker=0.081,
            rbt_kicker_thickness=0.005,
            rbt_kicker_width=0.08,
            rbt_wheel0_angle=45.0,
            rbt_wheel1_angle=135.0,
            rbt_wheel2_angle=225.0,
            rbt_wheel3_angle=315.0,
            rbt_radius=0.09,
            rbt_wheel_radius=0.028,
            rbt_motor_max_rpm=2000.0,
        )

        # Recalcula os valores que dependem do campo
        self.max_pos = max(
            self.field.width / 2, (self.field.length / 2) + self.field.penalty_length
        )
        max_wheel_rad_s = (self.field.rbt_motor_max_rpm / 60) * 2 * np.pi
        self.max_v = max_wheel_rad_s * self.field.rbt_wheel_radius
        self.max_w = np.rad2deg(self.max_v / 0.095)

        # Inicializa o restante como no SSLBaseEnv
        self.frame = None
        self.last_frame = None
        self.steps = 0
        self.sent_commands = None

        # Inicializa o renderizador

        self.field_renderer = SSLRenderField()
        self.window_surface = None
        self.window_size = self.field_renderer.window_size
        self.clock = None

        # Inicializa os espaços de ação e observação
        # Você precisa definir isso com base no seu caso de uso

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(3,),  # Exemplo: v_x, v_y, v_theta
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(11,),  # Exemplo: posição da bola, velocidade, posição do robô, etc.
            dtype=np.float32,
        )

    # Implemente os métodos abstratos necessários
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
            ],
            dtype=np.float32,
        )

    def _calculate_reward_and_done(self):
        """Calcula a recompensa e condição de término"""
        # Implemente sua lógica de recompensa aqui
        reward = 0
        done = False

        # Exemplo: terminar após um certo número de passos
        if self.steps >= 1200:  # 30 segundos a 40 FPS
            done = True

        return reward, done

    def _get_initial_positions_frame(self):
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

        # Posiciona os robôs azuis
        pos_frame.robots_blue = {}
        for i in range(self.n_robots_blue):
            while True:
                robot_x = np.random.uniform(-x_limit, x_limit)
                robot_y = np.random.uniform(-y_limit, y_limit)

                # Verifica se está longe o suficiente da bola
                distance_to_ball = np.sqrt(
                    (robot_x - pos_frame.ball.x) ** 2
                    + (robot_y - pos_frame.ball.y) ** 2
                )
                if distance_to_ball > 0.3:  # 30cm de distância mínima
                    break

            pos_frame.robots_blue[i] = Robot(
                yellow=False,
                id=i,
                x=robot_x,
                y=robot_y,
                theta=np.random.uniform(-180, 180),
            )

        # Posiciona os robôs amarelos
        pos_frame.robots_yellow = {}
        for i in range(self.n_robots_yellow):
            while True:
                robot_x = np.random.uniform(-x_limit, x_limit)
                robot_y = np.random.uniform(-y_limit, y_limit)

                # Verifica se está longe o suficiente da bola e dos robôs azuis
                distance_to_ball = np.sqrt(
                    (robot_x - pos_frame.ball.x) ** 2
                    + (robot_y - pos_frame.ball.y) ** 2
                )

                too_close_to_blue = False
                for blue_robot in pos_frame.robots_blue.values():
                    distance_to_blue = np.sqrt(
                        (robot_x - blue_robot.x) ** 2 + (robot_y - blue_robot.y) ** 2
                    )
                    if distance_to_blue < 0.3:  # 30cm de distância mínima
                        too_close_to_blue = True
                        break

                if distance_to_ball > 0.3 and not too_close_to_blue:
                    break

            pos_frame.robots_yellow[i] = Robot(
                yellow=True,
                id=i,
                x=robot_x,
                y=robot_y,
                theta=np.random.uniform(-180, 180),
            )

        return pos_frame
