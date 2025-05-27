from playgrounds.reach_to_point_el import SSLELReachToPoint, Position
import numpy as np
import pygame

class SSLELReachToPointAndTheta(SSLELReachToPoint):
    def __init__(self, render_mode=None, max_steps=1200):
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.target_theta = 0.0

    def _get_initial_positions_frame(self):
        pos_frame = super()._get_initial_positions_frame()
        self.target_theta = np.random.uniform(-np.pi, np.pi)
        return pos_frame

    def _frame_to_observations(self):
        obs = super()._frame_to_observations()
        robot = self.frame.robots_blue[0]

        return obs

    def _calculate_reward_and_done(self):
        reward, done = super()._calculate_reward_and_done()
        robot = self.frame.robots_blue[0]
        theta_error = np.abs(np.arctan2(np.sin(self.target_theta - np.deg2rad(robot.theta)),
                                        np.cos(self.target_theta - np.deg2rad(robot.theta))))
        success = (self.episodes_metrics['final_distance'] < 0.09) and (theta_error < 0.1)  # 0.1 rad ≈ 5.7 graus

        reward -= 5.0 * theta_error  

        self.episodes_metrics['success'] = success
        self.episodes_metrics['theta_error'] = theta_error

        if success:
            reward += 20.0  # ou outro valor

        return reward, done or success

    def _draw_target(self, target_pos, screen):
        # Chama o método original para desenhar o quadrado
        super()._draw_target(target_pos, screen)

        # Parâmetros do triângulo
        robot_radius = self.field.rbt_radius  # 0.09m
        scale = self.field_renderer.scale
        center_x, center_y = target_pos

        # Posição do centro do quadrado em pixels
        px = int(center_x * scale + self.field_renderer.center_x)
        py = int(center_y * scale + self.field_renderer.center_y)

        # Comprimento do triângulo (do centro até a ponta)
        tri_len = int(robot_radius * scale * 1.5)
        # Ângulo alvo (em radianos)
        theta = self.target_theta

        # Calcula os três vértices do triângulo
        # Ponta do triângulo (direção do target_theta)
        tip_x = px + int(tri_len * np.cos(theta))
        tip_y = py + int(tri_len * np.sin(theta))
        # Base do triângulo (duas pontas atrás)
        base_angle1 = theta + np.pi - np.pi/6
        base_angle2 = theta + np.pi + np.pi/6
        base1_x = px + int(robot_radius * scale * np.cos(base_angle1))
        base1_y = py + int(robot_radius * scale * np.sin(base_angle1))
        base2_x = px + int(robot_radius * scale * np.cos(base_angle2))
        base2_y = py + int(robot_radius * scale * np.sin(base_angle2))

        # Desenha o triângulo verde
        pygame.draw.polygon(
            screen,
            (0, 255, 0),  # Verde
            [(tip_x, tip_y), (base1_x, base1_y), (base2_x, base2_y)]
        )

    def _render(self):
        super()._render()
        self._draw_target([self.target_position.x, self.target_position.y], self.window_surface)