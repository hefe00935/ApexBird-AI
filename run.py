import pygame, random, numpy as np, torch, torch.nn as nn, os, math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Settings(Same As Train Or Won't Work) ---
SCREEN_W, SCREEN_H = 288, 512
FPS = 1000000
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
    def forward(self, x): return self.net(x)

class LegendBird:
    def __init__(self, model_path):
        self.brain = Brain()
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.brain.eval()
            print(f"🚀 Model Yüklendi: {model_path}")
        else:
            print(f"❌ HATA: {model_path} bulunamadı!")
            pygame.quit()
            exit()
            
        self.y = SCREEN_H // 2
        self.vel = 0
        self.score = 0
        self.done = False
        self.passed_pipe = False

    def get_state(self, pipes, current_gap):
        p1 = pipes[0]
        return np.array([
            self.y / SCREEN_H,
            self.vel / 15,
            (p1["x"] - 50) / SCREEN_W,
            (p1["gap_y"] + (current_gap/2) - self.y) / SCREEN_H,
            p1["gap_y"] / SCREEN_H,
            current_gap / 200,
            (256 - self.y) / 256
        ], dtype=np.float32)

def run_legend_mode(model_name):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("🏆 LEGEND TEST: FIXED SPEED MODE 🏆")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Impact", 22)

    bird = LegendBird(model_name)
    pipes = [{"x": SCREEN_W + 100, "gap_y": 200}]
    
    while not bird.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: bird.done = True

        # --- Hardness(Same as train!) ---
        curr_gap = max(MIN_GAP_LIMIT, 160 - (bird.score * 0.5))
        curr_spacing = max(280, 320 - (bird.score * 0.5))

        # Pipe Movement
        for p in pipes: 
            p["x"] -= FIXED_SPEED

        # Scor System(Boolean)
        if pipes[0]["x"] < 50 and not bird.passed_pipe:
            bird.score += 1
            bird.passed_pipe = True
            
        if pipes[0]["x"] < -52: 
            pipes.pop(0)
            bird.passed_pipe = False
        
        if pipes[-1]["x"] < SCREEN_W + 200:
            pipes.append({"x": pipes[-1]["x"] + curr_spacing, "gap_y": random.randint(120, 280)})

        state = bird.get_state(pipes, curr_gap)
        with torch.no_grad():
            action = bird.brain(torch.tensor(state).to(DEVICE)).argmax().item()
        
        if action == 1: bird.vel = JUMP
        bird.vel += GRAVITY
        bird.y += bird.vel

        # Collisions
        p = pipes[0]
        if bird.y < 0 or bird.y > SCREEN_H:
            bird.done = True
        if (p["x"] < 75 and p["x"] > 25 and (bird.y < p["gap_y"] or bird.y > p["gap_y"] + curr_gap)):
            bird.done = True

        # --- Render ---
        screen.fill((15, 15, 20)) 
        for p in pipes:
            # Pipes
            pygame.draw.rect(screen, (46, 204, 113), (p["x"], 0, 52, p["gap_y"]))
            pygame.draw.rect(screen, (46, 204, 113), (p["x"], p["gap_y"] + curr_gap, 52, SCREEN_H))
        
        # Apex Bird
        pygame.draw.circle(screen, (255, 255, 255), (50, int(bird.y)), 9)
        pygame.draw.circle(screen, (0, 255, 255), (50, int(bird.y)), 7, 2)
        
        # HUD
        screen.blit(font.render(f"MODEL: {model_name}", True, (200, 200, 200)), (10, SCREEN_H - 60))
        screen.blit(font.render(f"SCORE: {bird.score}", True, (255, 255, 255)), (10, SCREEN_H - 35))
        screen.blit(font.render(f"SPEED: {FIXED_SPEED}", True, (0, 255, 0)), (SCREEN_W - 100, SCREEN_H - 35))
        
        pygame.display.flip()
        clock.tick(FPS)

    print(f"🏁 Test Bitti! Final Skor: {bird.score}")
    pygame.quit()

if __name__ == "__main__":
    # Name of your model file please!
    MODEL_DOSYASI = "fixed_speed_s1748.pth" 
    run_legend_mode(MODEL_DOSYASI)