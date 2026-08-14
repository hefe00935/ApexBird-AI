# ApexBird-AI
> An AI that plays flappy bird.

ApexBird-AI is designed to produce models that outperform human capabilities. Using a custom Deep Neural Network (MLP) and a Genetic Algorithm, the agent achieves flawless navigation in high-speed, dynamic environments.

![Performance](https://img.shields.io/badge/Performance-Legendary-cyan)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Intelligence
The agent operates on a high-speed reward based flappy bird remake where it learns the physics and basics of playing then the 3 ones that does the best are selected apex-birds which teach other agents and so on.


---

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
python run.py

# Keyboard Controls (Critical to accidently close before saving)
# S  - Save the current "Elite" brain manually
#      • Snapshot: Saves the current best model as a .pth file immediately
#      • Logic: Capture perfect-flow moments before crash or next generation

# Testing Saved Models
# 1. Update the MODEL_PATH in test_model.py
# 2. Run the test
python test_model.py
