import pygame, random, numpy as np, torch, torch.nn as nn, copy, os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Parameters ---
POP_SIZE = 150
SCREEN_W, SCREEN_H = 288, 512
FPS = 1000 
GRAVITY = 0.4
JUMP = -6.0
FIXED_SPEED = 4.0 
MIN_GAP_LIMIT = 130 

class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        ).to(DEVICE)
        for m in self.net:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
    def forward(self, x): return self.net(x)

class Bird:
    def __init__(self, world, brain=None):
        self.world = world
        self.brain = brain if brain else Brain()
        self.reset()

    def reset(self):
        self.y = SCREEN_H // 2
        self.vel = 0
        self.score = 0 
        self.done = False
        self.fitness = 0
        self.passed_pipe = False

    def get_state(self, current_gap):
        p1 = self.world.pipes[0]
        return np.array([
            self.y / SCREEN_H,
            self.vel / 15,
            (p1["x"] - 50) / SCREEN_W,
            (p1["gap_y"] + (current_gap/2) - self.y) / SCREEN_H,
            p1["gap_y"] / SCREEN_H,
            current_gap / 200,
            (256 - self.y) / 256
        ], dtype=np.float32)

    def step(self, action, current_gap):
        if self.done: return
        if action == 1: self.vel = JUMP
        self.vel += GRAVITY
        self.y += self.vel
        
        p = self.world.pipes[0]
        center_y = p["gap_y"] + (current_gap / 2)
        
        if self.y < 0 or self.y > SCREEN_H:
            self.done = True
            self.fitness -= 5000 
            return

        if (p["x"] < 75 and p["x"] > 25 and (self.y < p["gap_y"] or self.y > p["gap_y"] + current_gap)):
            self.done = True
            self.fitness -= 25000 
            return

        if p["x"] < 50 and not self.passed_pipe:
            self.score += 1
            self.fitness += 150000 
            self.passed_pipe = True
        
        if p["x"] > 75: self.passed_pipe = False

        if p["x"] < 150:
            dist_to_center = abs(center_y - self.y)
            self.fitness += max(0, (120 - dist_to_center) / 8)
        
        self.fitness += 1

class ManualSaveTrainer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Impact", 20)
        self.gen = 1
        self.best_score_ever = 0
        self.birds = [Bird(self) for _ in range(POP_SIZE)]
        self.reset_pipes()

    def reset_pipes(self):
        self.pipes = [{"x": SCREEN_W + 100, "gap_y": random.randint(120, 280)}]

    def save_model(self, bird, tag="manual"):
        filename = f"model_{tag}_s{bird.score}_g{self.gen}.pth"
        torch.save(bird.brain.state_dict(), filename)
        print(f"\n💾 MODEL KAYDEDİLDİ: {filename}")

    def evolve(self):
        self.birds.sort(key=lambda x: x.fitness, reverse=True)
        if self.birds[0].score > self.best_score_ever:
            self.best_score_ever = self.birds[0].score
            if self.best_score_ever >= 200: self.save_model(self.birds[0], "auto")

        new_pop = []
        # Elite protection
        for i in range(15):
            new_pop.append(Bird(self, copy.deepcopy(self.birds[i].brain)))
        
        # Fresh dna
        for _ in range(int(POP_SIZE * 0.2)):
            new_pop.append(Bird(self))

        while len(new_pop) < POP_SIZE:
            parent = random.choice(self.birds[:12])
            child_brain = copy.deepcopy(parent.brain)
            with torch.no_grad():
                for param in child_brain.parameters():
                    if random.random() < 0.2:
                        param.add_(torch.randn(param.size()).to(DEVICE) * 0.015)
            new_pop.append(Bird(self, child_brain))
        self.birds = new_pop

    def run(self):
        while True:
            # Live Scor
            active_birds = [b for b in self.birds if not b.done]
            current_live_best = max((b.score for b in active_birds), default=0)
            
            # Keyboard Controls
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s: # 'S' to save the model
                        if active_birds:
                            best_active = max(active_birds, key=lambda x: x.fitness)
                            self.save_model(best_active, "manual")

            c_gap = max(MIN_GAP_LIMIT, 160 - (current_live_best * 0.5))
            c_spacing = max(280, 320 - (current_live_best * 0.5))

            for p in self.pipes: p["x"] -= FIXED_SPEED
            if self.pipes[0]["x"] < -52: self.pipes.pop(0)
            if self.pipes[-1]["x"] < SCREEN_W + 200:
                self.pipes.append({"x": self.pipes[-1]["x"] + c_spacing, "gap_y": random.randint(100, 320)})

            if not active_birds:
                self.evolve(); self.gen += 1; self.reset_pipes()
                for b in self.birds: b.reset()
                continue

            for b in active_birds:
                state = b.get_state(c_gap)
                with torch.no_grad():
                    action = b.brain(torch.tensor(state).to(DEVICE)).argmax().item()
                b.step(action, c_gap)

            self.render(active_birds, current_live_best, c_gap)
            if FPS <= 60: self.clock.tick(FPS)

    def render(self, active, live_score, gap):
        self.screen.fill((25, 25, 35))
        for p in self.pipes:
            pygame.draw.rect(self.screen, (70, 70, 80), (p["x"], 0, 52, p["gap_y"]))
            pygame.draw.rect(self.screen, (70, 70, 80), (p["x"], p["gap_y"] + gap, 52, SCREEN_H))
        for b in active:
            pygame.draw.circle(self.screen, (255, 255, 255), (50, int(b.y)), 7)
        
        # UI Panel
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 440, SCREEN_W, 72))
        self.screen.blit(self.font.render(f"GEN: {self.gen} | LIVE BEST: {live_score}", True, (255, 255, 0)), (10, 445))
        self.screen.blit(self.font.render(f"REKOR: {self.best_score_ever}", True, (0, 255, 0)), (10, 465))
        self.screen.blit(self.font.render(f"KAYDETMEK ICIN 'S' TUSUNA BAS", True, (200, 200, 200)), (10, 485))
        pygame.display.flip()

if __name__ == "__main__":
    ManualSaveTrainer().run()