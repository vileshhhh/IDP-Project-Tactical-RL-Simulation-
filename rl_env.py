"""
rl_env.py - Tactical Convoy Evasion System
==========================================
This file contains two main components:
  1. ConvoyEnv  - The 2D grid world (built with Pygame)
  2. QLearningAgent - The tabular Q-Learning AI brain

Designed for 2nd-semester engineering students.
Every major block is annotated so you can follow
the reinforcement-learning loop step by step.
"""

import pygame
import numpy as np
import random
import sys

# ──────────────────────────────────────────────
# GLOBAL CONSTANTS  (tweak these to experiment!)
# ──────────────────────────────────────────────
GRID_SIZE   = 12          # 12 × 12 cells
CELL_PX     = 54          # pixels per cell
WINDOW_W    = GRID_SIZE * CELL_PX
WINDOW_H    = GRID_SIZE * CELL_PX + 60   # +60 px for status bar

FPS         = 30          # frames per second during rendering

# Terrain codes stored in the grid array
TERRAIN_OPEN  = 0
TERRAIN_COVER = 1         # forests / buildings

# Movement cost (fuel penalty per step)
COST_OPEN  = 1            # -1 reward per step on open ground
COST_COVER = 2            # -2 reward per step through cover (rough terrain)

# Threat detection radius (cells)
THREAT_RADIUS = 2

# ──────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────
COLOR_BG        = (20,  25,  35)   # dark navy – background fill
COLOR_OPEN      = (45,  55,  70)   # slate – open terrain
COLOR_COVER     = (34,  80,  45)   # dark green – forest / cover
COLOR_BASE      = (220, 180,  30)  # gold – target base
COLOR_AGENT     = (50,  180, 230)  # cyan – convoy
COLOR_THREAT    = (220,  60,  60)  # red – patrol aircraft
COLOR_DANGER    = (220,  60,  60, 60)   # semi-transparent red overlay
COLOR_GRID_LINE = (30,  38,  55)   # subtle grid line
COLOR_TEXT      = (200, 210, 230)

# ──────────────────────────────────────────────
# HAND-CRAFTED MAP
# ──────────────────────────────────────────────
# 0 = Open, 1 = Cover (forest / building)
# The convoy starts at (0,0) top-left;
# the base is at (GRID_SIZE-1, GRID_SIZE-1) bottom-right.
RAW_MAP = [
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
]
# Convert to NumPy for fast indexing
TERRAIN_MAP = np.array(RAW_MAP, dtype=np.int8)

# ──────────────────────────────────────────────
# PATROL PATH – the aircraft follows this loop
# List of (row, col) waypoints; cycles endlessly.
# ──────────────────────────────────────────────
PATROL_PATH = [
    (0, 6), (0, 11), (3, 11), (6, 11),
    (11, 11), (11, 6), (11, 0), (6, 0),
    (0, 0),  (0, 6),
]


# ══════════════════════════════════════════════
# CLASS: ConvoyEnv
# The grid-world environment.  Follows the
# standard Gym-style API:  reset() → step()
# ══════════════════════════════════════════════
class ConvoyEnv:
    """
    2-D grid environment for the Tactical Convoy Evasion scenario.

    State tuple returned to the agent:
        (agent_row, agent_col, threat_row, threat_col, terrain_type)
    """

    def __init__(self, render: bool = True):
        self.render_mode = render
        self.grid        = TERRAIN_MAP.copy()

        # Fixed positions
        self.start_pos   = (0, 0)
        self.base_pos    = (GRID_SIZE - 1, GRID_SIZE - 1)

        # Patrol state
        self.patrol_path      = PATROL_PATH
        self.patrol_idx       = 0          # which waypoint we're heading toward
        self.patrol_progress  = 0.0        # 0.0 → 1.0 fraction between waypoints
        self.patrol_speed     = 0.08       # fraction of a segment per env step

        # Episode counters (set in reset)
        self.agent_pos  = self.start_pos
        self.step_count = 0
        self.max_steps  = 300

        # Pygame setup (only when rendering is requested)
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            pygame.display.set_caption("Tactical Convoy Evasion – RL Training")
            self.clock  = pygame.font.init() or pygame.time.Clock()
            self.clock  = pygame.time.Clock()
            self.font   = pygame.font.SysFont("consolas", 14)

    # ------------------------------------------------------------------
    def reset(self):
        """
        Reset the environment for a new episode.
        Returns the initial state tuple.
        """
        self.agent_pos       = self.start_pos
        self.step_count      = 0
        self.patrol_idx      = 0
        self.patrol_progress = 0.0
        return self._get_state()

    # ------------------------------------------------------------------
    def _get_threat_pos(self):
        """
        Compute the patrol aircraft's current (fractional) grid position
        by linearly interpolating between adjacent waypoints.
        Returns (row_float, col_float).
        """
        wp_a = self.patrol_path[self.patrol_idx % len(self.patrol_path)]
        wp_b = self.patrol_path[(self.patrol_idx + 1) % len(self.patrol_path)]
        t    = self.patrol_progress
        r    = wp_a[0] * (1 - t) + wp_b[0] * t
        c    = wp_a[1] * (1 - t) + wp_b[1] * t
        return (r, c)

    # ------------------------------------------------------------------
    def _advance_patrol(self):
        """Move the patrol aircraft one step along its route."""
        self.patrol_progress += self.patrol_speed
        if self.patrol_progress >= 1.0:
            self.patrol_progress = 0.0
            self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_path)

    # ------------------------------------------------------------------
    def _get_state(self):
        """
        Build the discrete state tuple the Q-table is indexed by.
        We round the threat position so the state space stays finite.
        """
        ar, ac          = self.agent_pos
        tr, tc          = self._get_threat_pos()
        tr_d, tc_d      = int(round(tr)), int(round(tc))   # discretise
        terrain         = int(self.grid[ar, ac])
        return (ar, ac, tr_d, tc_d, terrain)

    # ------------------------------------------------------------------
    def _distance_to_threat(self):
        """Euclidean distance between convoy and patrol aircraft."""
        ar, ac = self.agent_pos
        tr, tc = self._get_threat_pos()
        return np.sqrt((ar - tr) ** 2 + (ac - tc) ** 2)

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Apply an action and advance the world by one step.

        Actions:
            0 = Up    (row - 1)
            1 = Down  (row + 1)
            2 = Left  (col - 1)
            3 = Right (col + 1)
            4 = Hold  (don't move)

        Returns:
            next_state, reward, done, info_dict
        """
        ar, ac = self.agent_pos
        self.step_count += 1

        # ── Compute new position ──────────────────────────────────────
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        dr, dc = moves[action]
        nr, nc = ar + dr, ac + dc

        # Clamp to grid boundaries (wall bounce: stay put)
        nr = max(0, min(GRID_SIZE - 1, nr))
        nc = max(0, min(GRID_SIZE - 1, nc))
        self.agent_pos = (nr, nc)

        # ── Advance patrol aircraft ───────────────────────────────────
        self._advance_patrol()

        # ── Calculate reward ─────────────────────────────────────────
        reward = 0
        done   = False
        info   = {}

        terrain    = int(self.grid[nr, nc])
        dist       = self._distance_to_threat()
        near_threat = dist <= THREAT_RADIUS

        # 1. Reached the base → big positive reward
        if (nr, nc) == self.base_pos:
            reward += 100
            done    = True
            info["outcome"] = "success"

        # 2. In open terrain while threat is close → heavy penalty
        elif near_threat and terrain == TERRAIN_OPEN:
            reward -= 100
            done    = True
            info["outcome"] = "detected"

        # 3. Holding inside cover while threat is near → bonus
        elif near_threat and terrain == TERRAIN_COVER and action == 4:
            reward += 5
            info["hiding"] = True

        # 4. Per-step fuel/time cost (terrain dependent)
        if not done:
            step_cost = COST_COVER if terrain == TERRAIN_COVER else COST_OPEN
            reward   -= step_cost

        # 5. Timeout
        if self.step_count >= self.max_steps and not done:
            done = True
            info["outcome"] = "timeout"

        next_state = self._get_state()
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    def render(self, episode: int = 0, total_reward: float = 0):
        """
        Draw the current frame with Pygame.
        Call this once per step when you want to see the simulation.
        """
        if not self.render_mode:
            return

        # ── Pygame event pump (keep window responsive) ────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(COLOR_BG)

        # ── Draw terrain cells ────────────────────────────────────────
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x = c * CELL_PX
                y = r * CELL_PX
                color = COLOR_COVER if self.grid[r, c] == TERRAIN_COVER else COLOR_OPEN
                pygame.draw.rect(self.screen, color, (x, y, CELL_PX, CELL_PX))
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, (x, y, CELL_PX, CELL_PX), 1)

        # ── Draw threat danger zone (transparent overlay) ─────────────
        tr, tc   = self._get_threat_pos()
        danger_s = pygame.Surface((CELL_PX, CELL_PX), pygame.SRCALPHA)
        danger_s.fill((220, 60, 60, 55))
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if np.sqrt((r - tr) ** 2 + (c - tc) ** 2) <= THREAT_RADIUS:
                    self.screen.blit(danger_s, (c * CELL_PX, r * CELL_PX))

        # ── Draw base ─────────────────────────────────────────────────
        br, bc = self.base_pos
        bx, by = bc * CELL_PX, br * CELL_PX
        pygame.draw.rect(self.screen, COLOR_BASE, (bx + 6, by + 6, CELL_PX - 12, CELL_PX - 12), border_radius=4)
        lbl = self.font.render("BASE", True, (30, 25, 10))
        self.screen.blit(lbl, (bx + 8, by + 18))

        # ── Draw patrol aircraft ──────────────────────────────────────
        tx = int(tc * CELL_PX + CELL_PX // 2)
        ty = int(tr * CELL_PX + CELL_PX // 2)
        pygame.draw.circle(self.screen, COLOR_THREAT, (tx, ty), 14)
        pygame.draw.circle(self.screen, (255, 100, 100), (tx, ty), 14, 3)
        lbl = self.font.render("✈", True, (255, 255, 255))
        self.screen.blit(lbl, (tx - 7, ty - 8))

        # ── Draw agent (convoy) ───────────────────────────────────────
        ar, ac = self.agent_pos
        ax = ac * CELL_PX + CELL_PX // 2
        ay = ar * CELL_PX + CELL_PX // 2
        pygame.draw.circle(self.screen, COLOR_AGENT, (ax, ay), 12)
        pygame.draw.circle(self.screen, (100, 220, 255), (ax, ay), 12, 2)

        # ── Status bar ────────────────────────────────────────────────
        bar_y = GRID_SIZE * CELL_PX + 5
        status = (
            f"Episode: {episode}  |  "
            f"Steps: {self.step_count}  |  "
            f"Reward: {total_reward:.1f}  |  "
            f"Threat dist: {self._distance_to_threat():.1f}"
        )
        txt_surf = self.font.render(status, True, COLOR_TEXT)
        self.screen.blit(txt_surf, (8, bar_y))

        pygame.display.flip()
        self.clock.tick(FPS)

    # ------------------------------------------------------------------
    def close(self):
        if self.render_mode:
            pygame.quit()


# ══════════════════════════════════════════════
# CLASS: QLearningAgent
# Tabular Q-Learning with ε-greedy exploration
# ══════════════════════════════════════════════
class QLearningAgent:
    """
    Pure tabular Q-Learning agent.

    The Q-table is a Python dict (sparse) because the state space
    (12×12 agent) × (12×12 threat) × (2 terrain) = ~41,472 states
    is large but most states are never visited, so a dict is memory-
    efficient and still O(1) to look up.

    Q-Learning update rule (Bellman equation):
        Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]

    Where:
        α  (alpha)   = learning rate
        γ  (gamma)   = discount factor
        ε  (epsilon) = exploration probability
    """

    N_ACTIONS = 5   # Up, Down, Left, Right, Hold

    def __init__(
        self,
        alpha:   float = 0.15,   # learning rate
        gamma:   float = 0.95,   # discount factor
        epsilon: float = 1.0,    # starting exploration rate
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Sparse Q-table: state_tuple → np.array of shape (N_ACTIONS,)
        self.q_table: dict = {}

    # ------------------------------------------------------------------
    def _ensure_state(self, state):
        """
        Lazily initialise Q-values for a newly seen state to zeros.
        This avoids pre-allocating a giant multi-dim array.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.N_ACTIONS, dtype=np.float32)

    # ------------------------------------------------------------------
    def choose_action(self, state) -> int:
        """
        ε-greedy policy:
            With probability ε  → explore  (random action)
            With probability 1-ε → exploit (best known action)
        """
        self._ensure_state(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.N_ACTIONS - 1)
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    def update(self, state, action: int, reward: float, next_state, done: bool):
        """
        Apply the Q-Learning (Bellman) update for one (s, a, r, s') tuple.
        """
        self._ensure_state(state)
        self._ensure_state(next_state)

        current_q = self.q_table[state][action]

        # If the episode ended, there is no future value
        if done:
            target_q = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target_q    = reward + self.gamma * best_next_q

        # Bellman update
        self.q_table[state][action] += self.alpha * (target_q - current_q)

    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """
        Reduce exploration rate after each episode so the agent
        gradually shifts from 'try everything' to 'use what it learned'.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ══════════════════════════════════════════════
# TRAINING LOOP
# Called by app.py; yields progress via a
# shared data structure (status_dict).
# ══════════════════════════════════════════════
def run_training(
    n_episodes:    int  = 600,
    render:        bool = True,
    render_every:  int  = 50,      # show Pygame every N episodes
    status_dict:   dict = None,    # shared dict for Flask to read
):
    """
    Main training loop.

    Parameters
    ----------
    n_episodes    : total training episodes
    render        : whether to open a Pygame window at all
    render_every  : render one in every N episodes (keeps training fast)
    status_dict   : optional dict updated each episode so Flask can
                    stream progress to the browser

    Returns
    -------
    rewards_history : list of total reward per episode
    success_history : list of 1 (success) / 0 (not) per episode
    """
    env   = ConvoyEnv(render=render)
    agent = QLearningAgent()

    rewards_history = []
    success_history = []

    for ep in range(1, n_episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        done         = False

        # Decide whether to render this episode
        show = render and (ep % render_every == 0 or ep == 1)

        while not done:
            action                         = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state        = next_state
            total_reward += reward

            if show:
                env.render(episode=ep, total_reward=total_reward)

        # Record episode outcome
        rewards_history.append(total_reward)
        success = 1 if info.get("outcome") == "success" else 0
        success_history.append(success)

        # Decay exploration after every episode
        agent.decay_epsilon()

        # Update shared status dict (read by Flask)
        if status_dict is not None:
            status_dict["episode"]  = ep
            status_dict["reward"]   = total_reward
            status_dict["epsilon"]  = round(agent.epsilon, 4)
            status_dict["success"]  = success
            status_dict["running"]  = True

        # Console log every 50 episodes
        if ep % 50 == 0:
            win_rate = sum(success_history[-50:]) / 50 * 100
            print(
                f"[Ep {ep:>4}/{n_episodes}]  "
                f"Reward: {total_reward:>7.1f}  |  "
                f"ε: {agent.epsilon:.3f}  |  "
                f"Win-rate (last 50): {win_rate:.0f}%"
            )

    env.close()

    if status_dict is not None:
        status_dict["running"] = False
        status_dict["done"]    = True

    return rewards_history, success_history


# ──────────────────────────────────────────────
# Quick standalone test (python rl_env.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Running standalone training (600 episodes, render every 50)…")
    rewards, successes = run_training(n_episodes=600, render=True, render_every=50)
    print(f"\nTraining complete. Final success rate: {sum(successes[-100:]) / 100 * 100:.1f}%")
