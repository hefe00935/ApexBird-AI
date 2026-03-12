# 🦅 ApexBird-AI
> **High-Performance Autonomous Navigation Engine powered by Deep Reinforcement Learning.**

ApexBird-AI is an advanced autonomous flight system designed to outperform human capabilities in dynamic environments. Using a custom-built **Deep Neural Network (MLP)** and an **Evolutionary Genetic Algorithm**, the agent masters high-frequency decision-making and precise spatial navigation.

![Performance](https://img.shields.io/badge/Performance-Legendary-cyan)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🧠 The Intelligence
The agent operates on a 3-layer neural architecture, processing 7 spatial data points in real-time to maintain a "flow state" during flight.

### Neural Inputs (The 7 Senses):
* **Y-Altitude:** Normalized vertical position.
* **Velocity:** Instantaneous climbing/falling speed.
* **X-Proximity:** Horizontal distance to the next obstacle.
* **Gap Centering:** Vertical offset from the safe-zone center.
* **Ceiling Threshold:** Altitude of the upper pipe.
* **Dynamic Gap:** Real-time size of the passing window.
* **Upper Boundary:** Proximity to the screen ceiling.

## 🛠️ Advanced Features
* **Neuro-Reflex Controller:** Millisecond-latency inference via PyTorch.
* **Fixed-Speed Mastery:** Optimized for a constant speed of 4.0 to focus on pure spatial precision.
* **"New Blood" GA Logic:** Prevents population stagnation by injecting fresh genetic diversity during training plateaus.
* **Dynamic Reward Engineering:** Uses a "Magnet-Alignment" reward system to pull the agent toward the center of obstacles.

## 🚀 Installation & Usage

### 1. Clone the Repo
```bash
git clone [https://github.com/yourusername/ApexBird-AI.git](https://github.com/yourusername/ApexBird-AI.git)
cd ApexBird-AI