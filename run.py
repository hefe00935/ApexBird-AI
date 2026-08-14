import os
import pygame

import torch

from flappy_env import FPS, SCREEN_H, SCREEN_W, FlappyRenderer, FlappyWorld
from flappy_model import Brain, DEVICE


DEFAULT_MODEL = "model_best_s1991_g49.pth"


def load_brain(model_path):
    brain = Brain()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    brain.load_state_dict(torch.load(model_path, map_location=DEVICE))
    brain.eval()
    return brain


def choose_action(brain, state):
    with torch.no_grad():
        tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        return int(brain(tensor).argmax().item())


def run(model_path=DEFAULT_MODEL):
    pygame.init()
    pygame.display.set_caption("Flappy Bird")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    brain = load_brain(model_path)
    world = FlappyWorld()
    renderer = FlappyRenderer(screen)
    state = world.reset()
    running = True

    while running and world.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = choose_action(brain, state)
        state, _, _, _ = world.step(action == 1)
        renderer.draw(world, subtitle="AI pilot", extra=f"model: {model_path}")
        clock.tick(FPS)

    print(f"Final score: {world.score}")
    pygame.quit()


if __name__ == "__main__":
    run(os.environ.get("APEXBIRD_MODEL", DEFAULT_MODEL))
