import random
from dataclasses import dataclass

import numpy as np
import pygame


SCREEN_W = 288
SCREEN_H = 512
BASE_Y = 404
FPS = 30

BIRD_X = 57
BIRD_W = 34
BIRD_H = 24
BIRD_HITBOX_W = 28
BIRD_HITBOX_H = 20

PIPE_W = 52
PIPE_GAP = 100
PIPE_SPAWN_X = SCREEN_W + 10
PIPE_ADD_DISTANCE = SCREEN_W / 2
PIPE_VEL_X = -4

PLAYER_MAX_VEL_Y = 10
PLAYER_ACC_Y = 1
PLAYER_FLAP_ACC = -9

STATE_SIZE = 7
ACTION_SIZE = 2


@dataclass
class Pipe:
    x: float
    top: int
    bottom: int
    scored: bool = False


def _rand_pipe(rng=random):
    gap_y = rng.randrange(0, int(BASE_Y * 0.6 - PIPE_GAP))
    gap_y += int(BASE_Y * 0.2)
    return Pipe(float(PIPE_SPAWN_X), gap_y - PIPE_W, gap_y + PIPE_GAP)


class FlappyWorld:
    """Original mobile-style Flappy Bird dynamics without copyrighted assets."""

    def __init__(self, rng=None):
        self.rng = rng or random
        self.reset()

    def reset(self):
        self.player_y = int((SCREEN_H - BIRD_H) / 2)
        self.player_vel_y = -9
        self.player_flapped = False
        self.score = 0
        self.alive = True
        first = _rand_pipe(self.rng)
        second = _rand_pipe(self.rng)
        first.x = SCREEN_W + 200
        second.x = first.x + SCREEN_W / 2
        self.pipes = [first, second]
        return self.get_state()

    @property
    def bird_center_y(self):
        return self.player_y + BIRD_H / 2

    @property
    def bird_rect(self):
        x = BIRD_X + (BIRD_W - BIRD_HITBOX_W) / 2
        y = self.player_y + (BIRD_H - BIRD_HITBOX_H) / 2
        return pygame.Rect(round(x), round(y), BIRD_HITBOX_W, BIRD_HITBOX_H)

    def next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + PIPE_W >= BIRD_X:
                return pipe
        return self.pipes[0]

    def get_state(self):
        pipe = self.next_pipe()
        gap_top = pipe.bottom - PIPE_GAP
        gap_center = pipe.bottom - PIPE_GAP / 2
        return np.array(
            [
                self.bird_center_y / BASE_Y,
                self.player_vel_y / PLAYER_MAX_VEL_Y,
                (pipe.x + PIPE_W - BIRD_X) / SCREEN_W,
                (gap_center - self.bird_center_y) / BASE_Y,
                gap_top / BASE_Y,
                PIPE_GAP / BASE_Y,
                (BASE_Y - self.bird_center_y) / BASE_Y,
            ],
            dtype=np.float32,
        )

    def step(self, flap):
        if not self.alive:
            return self.get_state(), 0.0, True, {"score": self.score}

        reward = 0.1
        if flap and self.player_y > -2 * BIRD_H:
            self.player_vel_y = PLAYER_FLAP_ACC
            self.player_flapped = True

        if self.player_vel_y < PLAYER_MAX_VEL_Y and not self.player_flapped:
            self.player_vel_y += PLAYER_ACC_Y
        if self.player_flapped:
            self.player_flapped = False

        self.player_y += min(self.player_vel_y, BASE_Y - self.player_y - BIRD_H)
        if self.player_y < 0:
            self.player_y = 0

        for pipe in self.pipes:
            pipe.x += PIPE_VEL_X

        if self.pipes and self.pipes[0].x < -PIPE_W:
            self.pipes.pop(0)

        if self.pipes[-1].x <= PIPE_ADD_DISTANCE:
            self.pipes.append(_rand_pipe(self.rng))

        bird_mid = BIRD_X + BIRD_W / 2
        for pipe in self.pipes:
            pipe_mid = pipe.x + PIPE_W / 2
            if pipe_mid <= bird_mid < pipe_mid + abs(PIPE_VEL_X) and not pipe.scored:
                pipe.scored = True
                self.score += 1
                reward += 10.0

        if self.collides():
            self.alive = False
            reward -= 10.0

        return self.get_state(), reward, not self.alive, {"score": self.score}

    def collides(self):
        bird = self.bird_rect
        if bird.top <= 0 or bird.bottom >= BASE_Y:
            return True
        for pipe in self.pipes:
            upper = pygame.Rect(round(pipe.x), 0, PIPE_W, pipe.bottom - PIPE_GAP)
            lower = pygame.Rect(round(pipe.x), pipe.bottom, PIPE_W, BASE_Y - pipe.bottom)
            if bird.colliderect(upper) or bird.colliderect(lower):
                return True
        return False


class FlappyRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.frame = 0
        self.assets = self._build_assets()

    def _build_assets(self):
        sky = pygame.Surface((SCREEN_W, SCREEN_H))
        for y in range(SCREEN_H):
            mix = y / SCREEN_H
            color = (int(112 + 18 * mix), int(197 + 24 * mix), int(206 + 10 * mix))
            pygame.draw.line(sky, color, (0, y), (SCREEN_W, y))

        bird_frames = []
        for wing in (-5, 0, 5):
            surf = pygame.Surface((BIRD_W, BIRD_H), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (248, 222, 74), (1, 2, 29, 20))
            pygame.draw.ellipse(surf, (255, 244, 145), (7, 9, 17, 9))
            pygame.draw.polygon(surf, (236, 167, 47), [(21, 12), (34, 8), (34, 16)])
            pygame.draw.circle(surf, (255, 255, 255), (24, 7), 5)
            pygame.draw.circle(surf, (24, 24, 24), (26, 7), 2)
            pygame.draw.ellipse(surf, (241, 188, 54), (5, 8 + wing, 15, 9))
            bird_frames.append(surf)

        pipe = pygame.Surface((PIPE_W, BASE_Y), pygame.SRCALPHA)
        pygame.draw.rect(pipe, (83, 190, 64), (4, 0, PIPE_W - 8, BASE_Y))
        pygame.draw.rect(pipe, (120, 220, 82), (8, 0, 9, BASE_Y))
        pygame.draw.rect(pipe, (48, 138, 54), (PIPE_W - 12, 0, 7, BASE_Y))
        pygame.draw.rect(pipe, (45, 105, 40), (0, 0, PIPE_W, BASE_Y), 2)

        base = pygame.Surface((SCREEN_W + 48, SCREEN_H - BASE_Y))
        base.fill((222, 216, 149))
        pygame.draw.rect(base, (222, 184, 91), (0, 0, SCREEN_W + 48, 14))
        for x in range(-24, SCREEN_W + 48, 24):
            pygame.draw.polygon(base, (190, 157, 74), [(x, 14), (x + 12, 28), (x + 24, 14)])
        return {"sky": sky, "bird": bird_frames, "pipe": pipe, "base": base}

    def draw(self, world, subtitle=None, extra=None, flock_worlds=None):
        self.frame += 1
        self.screen.blit(self.assets["sky"], (0, 0))

        for pipe in world.pipes:
            x = round(pipe.x)
            top_height = pipe.bottom - PIPE_GAP
            upper = pygame.transform.flip(self.assets["pipe"], False, True)
            self.screen.blit(upper, (x, top_height - BASE_Y))
            self.screen.blit(self.assets["pipe"], (x, pipe.bottom))

        base_shift = (self.frame * abs(PIPE_VEL_X)) % 24
        self.screen.blit(self.assets["base"], (-base_shift, BASE_Y))

        if flock_worlds:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            for flock_world in flock_worlds:
                if flock_world is world:
                    continue
                pygame.draw.circle(
                    overlay,
                    (255, 244, 145, 70),
                    (round(BIRD_X + BIRD_W / 2), round(flock_world.bird_center_y)),
                    3,
                )
            self.screen.blit(overlay, (0, 0))

        frame = self.assets["bird"][(self.frame // 5) % len(self.assets["bird"])]
        angle = max(-25, min(25, -world.player_vel_y * 3))
        rotated = pygame.transform.rotate(frame, angle)
        center = (BIRD_X + BIRD_W / 2, world.bird_center_y)
        self.screen.blit(rotated, rotated.get_rect(center=center))

        score = self.font.render(str(world.score), True, (255, 255, 255))
        shadow = self.font.render(str(world.score), True, (65, 65, 65))
        self.screen.blit(shadow, shadow.get_rect(center=(SCREEN_W / 2 + 1, 43)))
        self.screen.blit(score, score.get_rect(center=(SCREEN_W / 2, 42)))

        if subtitle:
            label = self.small_font.render(subtitle, True, (255, 255, 255))
            self.screen.blit(label, (8, 8))
        if extra:
            label = self.small_font.render(extra, True, (255, 255, 255))
            self.screen.blit(label, (8, SCREEN_H - 22))

        pygame.display.flip()
