import argparse
import copy
import random
import warnings

import numpy as np
import pygame

warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*", category=FutureWarning)

import torch
import torch.nn.functional as F

from flappy_env import FPS, PIPE_GAP, SCREEN_H, SCREEN_W, FlappyRenderer, FlappyWorld
from flappy_model import Brain, DEVICE


POP_SIZE = 500
ELITE_COUNT = 25
IMMIGRATION_RATE = 0.12
MUTATION_RATE = 0.12
MUTATION_SCALE = 0.012
TRAIN_FPS = 600
SHOW_FLOCK = False


def require_training_device(require_cuda):
    if require_cuda and DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA GPU was requested, but PyTorch cannot see one. "
            "Install a CUDA-enabled torch build or run with --no-require-cuda."
        )
    print(f"Using torch device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")


def expert_action(world):
    pipe = world.next_pipe()
    gap_center = pipe.bottom - PIPE_GAP / 2
    pipe_distance = pipe.x - 57
    lead = max(0.0, min(38.0, pipe_distance * 0.14))
    target_y = gap_center - lead
    return int(world.player_vel_y > -2 and world.bird_center_y > target_y)


def warm_start_brain(samples=30_000, epochs=5, batch_size=1024, lr=1e-3):
    brain = Brain()
    states = []
    labels = []

    while len(states) < samples:
        world = FlappyWorld()
        for _ in range(random.randrange(4, 150)):
            if not world.alive:
                break
            if random.random() < 0.12:
                world.step(True)
            else:
                world.step(False)
        if world.alive:
            states.append(world.get_state())
            labels.append(expert_action(world))

    x = torch.tensor(np.asarray(states), dtype=torch.float32, device=DEVICE)
    y = torch.tensor(labels, dtype=torch.long, device=DEVICE)
    optimizer = torch.optim.AdamW(brain.parameters(), lr=lr, weight_decay=1e-4)

    brain.train()
    for _ in range(epochs):
        order = torch.randperm(x.size(0), device=DEVICE)
        for start in range(0, x.size(0), batch_size):
            batch = order[start : start + batch_size]
            loss = F.cross_entropy(brain(x[batch]), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    brain.eval()
    return brain


class Agent:
    def __init__(self, brain=None, seed=None):
        self.brain = brain if brain is not None else Brain()
        self.reset(seed)

    def reset(self, seed=None):
        rng = random.Random(seed) if seed is not None else None
        self.world = FlappyWorld(rng)
        self.state = self.world.get_state()
        self.fitness = 0.0
        self.frames = 0

    @property
    def alive(self):
        return self.world.alive

    @property
    def score(self):
        return self.world.score

    def step(self, action):
        self.state, reward, done, _ = self.world.step(action == 1)
        self.frames += 1

        pipe = self.world.next_pipe()
        gap_center = pipe.bottom - PIPE_GAP / 2
        center_penalty = abs(gap_center - self.world.bird_center_y) / 120
        survival_bonus = max(0.0, 1.0 - center_penalty)
        self.fitness += reward + survival_bonus
        if done:
            self.fitness += self.score * 1000 + self.frames * 0.1


class Trainer:
    def __init__(self, args):
        self.args = args
        require_training_device(args.require_cuda)
        self.render = not args.headless
        if self.render:
            pygame.init()
            pygame.display.set_caption("ApexBird Trainer")
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()
            self.renderer = FlappyRenderer(self.screen)
        else:
            self.screen = None
            self.clock = None
            self.renderer = None

        self.generation = 1
        self.training_steps = 0
        self.best_score = 0
        self.generation_seed = random.randrange(1_000_000_000)
        base_brain = warm_start_brain(args.warmup_samples, args.warmup_epochs) if args.warmup else None
        self.agents = self.build_initial_population(base_brain)

    def _new_agent(self, base_brain=None, seed=None):
        brain = copy.deepcopy(base_brain) if base_brain is not None else Brain()
        return Agent(brain, seed)

    def build_initial_population(self, base_brain):
        if base_brain is None:
            return [Agent(seed=self.generation_seed) for _ in range(self.args.population)]

        population = [Agent(copy.deepcopy(base_brain), self.generation_seed)]
        immigrants = int(self.args.population * self.args.immigration)
        population.extend(Agent(seed=self.generation_seed) for _ in range(immigrants))

        while len(population) < self.args.population:
            brain = copy.deepcopy(base_brain)
            self.mutate(brain)
            population.append(Agent(brain, self.generation_seed))
        return population

    def save_model(self, agent, tag="manual"):
        filename = f"model_{tag}_s{agent.score}_g{self.generation}.pth"
        torch.save(agent.brain.state_dict(), filename)
        print(f"Saved model: {filename}")

    def mutate(self, brain):
        with torch.no_grad():
            for param in brain.parameters():
                mask = torch.rand(param.size(), device=DEVICE) < self.args.mutation_rate
                noise = torch.randn(param.size(), device=DEVICE) * self.args.mutation_scale
                param.add_(mask * noise)

    def evolve(self):
        self.agents.sort(key=lambda agent: agent.fitness, reverse=True)
        champion = self.agents[0]
        if champion.score > self.best_score:
            self.best_score = champion.score
            self.save_model(champion, "best")

        print(
            f"gen {self.generation:04d} score {champion.score:04d} "
            f"best {self.best_score:04d} fitness {champion.fitness:.1f}"
        )

        next_seed = random.randrange(1_000_000_000)
        elite_count = min(self.args.elites, len(self.agents))
        next_population = [Agent(copy.deepcopy(agent.brain), next_seed) for agent in self.agents[:elite_count]]
        immigrants = int(self.args.population * self.args.immigration)
        next_population.extend(Agent(seed=next_seed) for _ in range(immigrants))

        while len(next_population) < self.args.population:
            parent = random.choice(self.agents[:elite_count])
            child_brain = copy.deepcopy(parent.brain)
            self.mutate(child_brain)
            next_population.append(Agent(child_brain, next_seed))

        self.generation += 1
        self.generation_seed = next_seed
        self.agents = next_population

    def choose_actions(self, active):
        states = torch.as_tensor(
            np.asarray([agent.state for agent in active], dtype=np.float32),
            dtype=torch.float32,
            device=DEVICE,
        )
        with torch.inference_mode():
            logits = self.forward_population(active, states)
            return logits.argmax(dim=1).cpu().tolist()

    def forward_population(self, active, states):
        weights = [[layer.weight for layer in agent.brain.net if isinstance(layer, torch.nn.Linear)] for agent in active]
        biases = [[layer.bias for layer in agent.brain.net if isinstance(layer, torch.nn.Linear)] for agent in active]

        x = states.unsqueeze(2)
        x = torch.bmm(torch.stack([agent_weights[0] for agent_weights in weights]), x).squeeze(2)
        x = torch.relu(x + torch.stack([agent_biases[0] for agent_biases in biases]))
        x = torch.bmm(torch.stack([agent_weights[1] for agent_weights in weights]), x.unsqueeze(2)).squeeze(2)
        x = torch.relu(x + torch.stack([agent_biases[1] for agent_biases in biases]))
        x = torch.bmm(torch.stack([agent_weights[2] for agent_weights in weights]), x.unsqueeze(2)).squeeze(2)
        return x + torch.stack([agent_biases[2] for agent_biases in biases])

    def maybe_draw(self, active):
        if not self.render:
            return
        leader = max(active, key=lambda agent: (agent.score, agent.fitness))
        self.renderer.draw(
            leader.world,
            subtitle=f"gen {self.generation} alive {len(active)}/{self.args.population}",
            extra=f"best {self.best_score}  fps {self.args.fps or 'max'}  S saves",
            flock_worlds=[agent.world for agent in active] if self.args.show_flock else None,
        )
        if self.args.fps:
            self.clock.tick(self.args.fps)

    def pump_events(self, active):
        if not self.render:
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s and active:
                self.save_model(max(active, key=lambda agent: agent.fitness))
        return True

    def run(self):
        running = True
        while running:
            active = [agent for agent in self.agents if agent.alive]
            running = self.pump_events(active)
            if not running:
                break
            if not active:
                self.evolve()
                if self.args.generations and self.generation > self.args.generations:
                    break
                if self.args.target_score and self.best_score >= self.args.target_score:
                    break
                continue

            for agent, action in zip(active, self.choose_actions(active)):
                agent.step(action)
            self.training_steps += 1
            if self.training_steps % self.args.draw_every == 0:
                self.maybe_draw(active)

        if self.render:
            pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Flappy Bird-style neural pilot.")
    parser.add_argument("--headless", action="store_true", help="train without opening a Pygame window")
    parser.add_argument("--population", type=int, default=POP_SIZE)
    parser.add_argument("--elites", type=int, default=ELITE_COUNT)
    parser.add_argument("--immigration", type=float, default=IMMIGRATION_RATE)
    parser.add_argument("--mutation-rate", type=float, default=MUTATION_RATE)
    parser.add_argument("--mutation-scale", type=float, default=MUTATION_SCALE)
    parser.add_argument("--fps", type=int, default=TRAIN_FPS, help="0 means uncapped")
    parser.add_argument("--draw-every", type=int, default=4, help="render every N training steps")
    parser.add_argument("--generations", type=int, default=0, help="0 means run until stopped")
    parser.add_argument("--target-score", type=int, default=0, help="0 disables the target")
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-samples", type=int, default=30_000)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--show-flock", action=argparse.BooleanOptionalAction, default=SHOW_FLOCK)
    args = parser.parse_args()
    if args.draw_every < 1:
        parser.error("--draw-every must be at least 1")
    return args


if __name__ == "__main__":
    Trainer(parse_args()).run()
