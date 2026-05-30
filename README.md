# Reinforcement Learning-Based AI Solution for Constructive Simulation of Ground Agents in Tactical Scenarios

This repository contains the working prototype for our Interdisciplinary Project (1BPRJ208) developed during the 2nd semester at BMS Institute of Technology & Management. The framework explores shifting from rigid, rule-based Computer Generated Forces (CGF) to adaptive, non-deterministic Reinforcement Learning agents.

## 🎯 Project Framework
Traditional training simulations rely heavily on scripted, hard-coded "If-Then" logic, which makes adversarial entities entirely predictable. This project implements a model-free **Tabular Q-Learning** engine inside a customized grid environment to train a tactical ground agent (such as an infantry unit or a convoy vehicle) to dynamically navigate terrain, exploit cover assets, and actively minimize risk from a patrolling threat.

### System Architecture Breakdown
1. **Simulation Environment:** A discretised $N \times N$ grid with dynamic terrain values mapping "Open Terrain" vs "Tactical Cover" locations.
2. **RL Agent Engine:** Implements the classic temporal-difference Bellman update rule with decaying epsilon-greedy exploration.
3. **Control Interface:** A lightweight, interactive web-driven dashboard built on Flask to easily configure parameters, launch the simulation windows, and log training metrics.

## ⚙️ Core Technical Stack
* **Language Environment:** Python 3.10+
* **Mathematics & Optimization Matrix:** NumPy
* **Visual Rendering Loop:** Pygame
* **Web UI Wrapper Framework:** Flask
* **Performance Plot Logging:** Matplotlib

## 📂 Repository Structure
```text
├── app.py              # Main Flask server application & browser control endpoints
├── rl_env.py           # Core Pygame environment matrix and Q-learning training iterations
├── requirements.txt    # List of open-source library dependencies
└── templates/
    └── index.html      # Frontend HTML control panel interface dashboard
