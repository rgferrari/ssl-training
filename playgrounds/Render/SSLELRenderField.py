# playgrounds/Render/SSLELRenderField.py
import pygame
from rsoccer_gym.Render.field import SSLRenderField, Sim2DRenderField
from rsoccer_gym.Render.utils import COLORS


class SSLELRenderField(SSLRenderField):
    """
    Renderizador personalizado para o campo EL
    Ajusta as dimensões para corresponder ao campo personalizado
    """

    # Estas dimensões devem corresponder ao campo personalizado em SSLELSim.py
    length = 4.5  # Comprimento do campo em metros
    width = 3.0  # Largura do campo em metros
    margin = 0.35
    center_circle_r = 0.75
    penalty_length = 0.5
    penalty_width = 1.35
    goal_area_length = 0
    goal_area_width = 0
    goal_width = 0.8
    goal_depth = 0.18
    corner_arc_r = 0.01
    _scale = 100  # Escala para renderização (ajuste conforme necessário)

    def pos_to_screen(self, x, y):
        """
        Converte uma posição do mundo (metros) para uma posição na tela (pixels).

        Args:
            x (float): Coordenada x no mundo (metros).
            y (float): Coordenada y no mundo (metros).

        Returns:
            tuple: Uma tupla (x_s, y_s) com as coordenadas em pixels.
        """
        # self.center_x e self.center_y são inicializados na classe pai SSLRenderField
        x_s = self.center_x + x * self.scale
        y_s = self.center_y - y * self.scale  # O eixo Y do Pygame é invertido (cresce para baixo)
        return x_s, y_s





if __name__ == "__main__":
    field = Sim2DRenderField()
    pygame.display.init()
    pygame.display.set_caption("SSL Environment")
    window = pygame.display.set_mode(field.window_size)
    clock = pygame.time.Clock()
    while True:
        field.draw(window)
        pygame.event.pump()
        pygame.display.update()
        clock.tick(60)
