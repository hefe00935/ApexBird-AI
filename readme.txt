# 🦅 ApexBird-AI
## High-Performance Autonomous Navigation Engine
> Mastering spatial precision through Deep Reinforcement Learning and Evolutionary Algorithms.

ApexBird-AI is an advanced autonomous flight system designed to outperform human capabilities. Using a custom Deep Neural Network (MLP) and a Genetic Algorithm, the agent achieves flawless navigation in high-speed, dynamic environments.

![Performance](https://img.shields.io/badge/Performance-Legendary-cyan)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Intelligence
The agent operates on a high-speed 3-layer neural architecture. It processes **7 Core Spatial Senses** in real-time to maintain a perfect "flow state."

### The 7 Neural Inputs
1. **Y-Altitude:** Normalized vertical position
2. **Velocity:** Instantaneous climbing/falling speed
3. **X-Proximity:** Horizontal distance to the next obstacle
4. **Gap Centering:** Vertical offset from the safe-zone center
5. **Ceiling Threshold:** Altitude of the upper pipe
6. **Dynamic Gap:** Real-time size of the passing window
7. **Upper Boundary:** Proximity to the screen ceiling

---

## 🛠️ Advanced Features
- **Neuro-Reflex Controller:** Millisecond-latency inference via PyTorch
- **Fixed-Speed Mastery:** Locked at Speed 4.0 for pure spatial precision
- **"New Blood" GA Logic:** Injects fresh genetic diversity (20–50% immigration rate) to bypass training plateaus
- **Magnet-Alignment:** A dynamic reward system that pulls the agent toward the gap center

---

## 🚀 Installation, Training & Testing

```bash
# 1. Clone the Repository
git clone https://github.com/hefe00935/ApexBird-AI.git
cd ApexBird-AI

# 2. Install Dependencies
pip install torch numpy pygame

# 3. Start Training
python train.py

# ⌨️ Keyboard Controls (Critical)
# S  - Save the current "Elite" brain manually
#      • Instant Snapshot: Saves the current best model as a .pth file immediately
#      • Manual Logic: Capture perfect-flow moments before crash or next generation

# 🏆 Testing Saved Models
# 1. Update the MODEL_PATH in test_model.py
# 2. Run the test
python test_model.py
