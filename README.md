# Reinforcement Learning-Based AI Solution for Constructive Simulation of Ground Agents in Tactical Scenarios

[cite_start]This repository contains the working prototype for our Interdisciplinary Project (1BPRJ208) developed during the 2nd semester at BMS Institute of Technology & Management[cite: 5, 7, 8, 9, 10]. [cite_start]The framework explores shifting from rigid, rule-based Computer Generated Forces (CGF) to adaptive, non-deterministic Reinforcement Learning agents[cite: 40, 49].

## 🎯 Project Framework
[cite_start]Traditional training simulations rely heavily on scripted, hard-coded "If-Then" logic, which makes adversarial entities entirely predictable[cite: 49]. [cite_start]This project implements a model-free **Tabular Q-Learning** engine inside a customized grid environment to train a tactical ground agent (such as an infantry unit or a convoy vehicle) to dynamically navigate terrain, exploit cover assets, and actively minimize risk from a patrolling threat[cite: 39, 42, 211, 213, 218].

### System Architecture Breakdown
1. [cite_start]**Simulation Environment:** A discretised $N \times N$ grid with dynamic terrain values mapping "Open Terrain" vs "Tactical Cover" locations[cite: 251, 254].
2. [cite_start]**RL Agent Engine:** Implements the classic temporal-difference Bellman update rule with decaying epsilon-greedy exploration[cite: 254, 255].
3. [cite_start]**Control Interface:** A lightweight, interactive web-driven dashboard built on Flask to easily configure parameters, launch the simulation windows, and log training metrics[cite: 256, 257, 291].

## ⚙️ Core Technical Stack
* [cite_start]**Language Environment:** Python 3.10+ [cite: 327]
* [cite_start]**Mathematics & Optimization Matrix:** NumPy [cite: 327]
* [cite_start]**Visual Rendering Loop:** Pygame [cite: 329]
* [cite_start]**Web UI Wrapper Framework:** Flask 
* [cite_start]**Performance Plot Logging:** Matplotlib [cite: 329]

## 📂 Repository Structure
```text
├── app.py              # Main Flask server application & browser control endpoints
├── rl_env.py           # Core Pygame environment matrix and Q-learning training iterations
├── requirements.txt    # List of open-source library dependencies
└── templates/
    └── index.html      # Frontend HTML control panel interface dashboard
