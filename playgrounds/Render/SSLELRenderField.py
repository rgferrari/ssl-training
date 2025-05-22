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
    _scale = 500  # Escala para renderização (ajuste conforme necessário)


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
