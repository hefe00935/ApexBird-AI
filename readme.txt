# 🦅 ApexBird-AI
## **High-Performance Autonomous Navigation Engine**
> **Mastering spatial precision through Deep Reinforcement Learning and Evolutionary Algorithms.**

ApexBird-AI is an advanced autonomous flight system designed to outperform human capabilities. Using a custom **Deep Neural Network (MLP)** and a **Genetic Algorithm**, the agent achieves flawless navigation in high-speed, dynamic environments.

![Performance](https://img.shields.io/badge/Performance-Legendary-cyan)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 🧠 THE INTELLIGENCE
The agent operates on a high-speed 3-layer neural architecture. It processes **7 Core Spatial Senses** in real-time to maintain a perfect "flow state."

### **The 7 Neural Inputs:**
1. **Y-Altitude:** Normalized vertical position.
2. **Velocity:** Instantaneous climbing/falling speed.
3. **X-Proximity:** Horizontal distance to the next obstacle.
4. **Gap Centering:** Vertical offset from the safe-zone center.
5. **Ceiling Threshold:** Altitude of the upper pipe.
6. **Dynamic Gap:** Real-time size of the passing window.
7. **Upper Boundary:** Proximity to the screen ceiling.

---

# 🛠️ ADVANCED FEATURES
* **Neuro-Reflex Controller:** Millisecond-latency inference via PyTorch.
* **Fixed-Speed Mastery:** Locked at **Speed 4.0** for pure spatial precision.
* **"New Blood" GA Logic:** Injects fresh genetic diversity (20-50% immigration rate) to bypass training plateaus.
* **Magnet-Alignment:** A dynamic reward system that pulls the agent toward the gap center.

---

# 🚀 INSTALLATION & USAGE

### **1. Clone the Repository**
```bash
git clone [https://github.com/hefe00935/ApexBird-AI.git](https://github.com/hefe00935/ApexBird-AI.git)
cd ApexBird-AI

2. Install Dependencies
Bash

pip install torch numpy pygame

3. Start Training
Bash

python train.py

⌨️ KEYBOARD CONTROLS (CRITICAL)
Press S to SAVE THE BRAIN

While the training window is active, you can manually record the current "Elite" agent.

    Instant Snapshot: Pressing S saves the current best model as a .pth file immediately.

    Manual Logic: This allows you to capture "Perfect Flow" moments manually before the next generation starts or a sudden crash occurs.

🏆 TESTING THE LEGENDS

To watch your saved models (like the 1748+ score elites) fly in a synced environment:

    Update the MODEL_PATH in test_model.py.

    Run the command:

Bash

python test_model.py

📜 LICENSE

Distributed under the MIT License.

ApexBird-AI: Pushing the limits of autonomous agent navigation.
