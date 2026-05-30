"""
app.py - Flask Control Panel for Tactical Convoy Evasion
=========================================================
This Flask server acts as the project dashboard.

Routes
------
GET  /              → Renders the HTML control panel
POST /train         → Starts training in a background thread
GET  /status        → JSON feed of live training progress
GET  /plots         → Generates + serves Matplotlib plots as base64
GET  /results       → JSON of final metrics after training

Architecture note
-----------------
Training runs in a daemon Thread so Flask can keep responding
to the browser while Pygame/Q-Learning runs in parallel.
A shared dict `TRAINING_STATUS` is the communication channel
(thread-safe for simple reads/writes in CPython because of the GIL).
"""

from flask import Flask, render_template, jsonify, request
import threading
import base64
import io
import os

# We import only the training runner here; Pygame is initialised
# inside run_training when it actually runs.
from rl_env import run_training

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed for Flask)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ──────────────────────────────────────────────
# Flask app setup
# ──────────────────────────────────────────────
app = Flask(__name__)

# Ensure the templates folder exists alongside this file
os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)

# ──────────────────────────────────────────────
# Shared mutable state between threads
# ──────────────────────────────────────────────
TRAINING_STATUS = {
    "running":  False,   # True while training loop is active
    "done":     False,   # True after training completes
    "episode":  0,
    "reward":   0.0,
    "epsilon":  1.0,
    "success":  0,
}

# These hold the full history after training finishes
REWARDS_HISTORY  = []
SUCCESS_HISTORY  = []

# Lock so only one training run at a time
_training_lock = threading.Lock()


# ══════════════════════════════════════════════
# BACKGROUND TRAINING THREAD
# ══════════════════════════════════════════════
def _training_worker(n_episodes: int, render: bool, render_every: int):
    """
    Wrapper run inside a thread.
    Calls run_training() and stores results in module-level lists.
    """
    global REWARDS_HISTORY, SUCCESS_HISTORY

    rewards, successes = run_training(
        n_episodes   = n_episodes,
        render       = render,
        render_every = render_every,
        status_dict  = TRAINING_STATUS,   # live progress feed
    )

    REWARDS_HISTORY  = rewards
    SUCCESS_HISTORY  = successes


# ══════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the HTML dashboard."""
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    """
    Start training in a background thread.
    Accepts JSON body:
        { "episodes": 600, "render": true, "render_every": 50 }
    """
    global REWARDS_HISTORY, SUCCESS_HISTORY

    # Reject if already running
    if TRAINING_STATUS["running"]:
        return jsonify({"status": "error", "message": "Training already in progress."}), 409

    # Parse request parameters (with safe defaults)
    data         = request.get_json(silent=True) or {}
    n_episodes   = int(data.get("episodes",    600))
    render       = bool(data.get("render",     True))
    render_every = int(data.get("render_every", 50))

    # Reset status flags
    TRAINING_STATUS.update({
        "running": True,
        "done":    False,
        "episode": 0,
        "reward":  0.0,
        "epsilon": 1.0,
        "success": 0,
    })
    REWARDS_HISTORY.clear()
    SUCCESS_HISTORY.clear()

    # Launch training in daemon thread
    t = threading.Thread(
        target=_training_worker,
        args=(n_episodes, render, render_every),
        daemon=True,
    )
    t.start()

    return jsonify({"status": "started", "episodes": n_episodes})


@app.route("/status")
def status():
    """
    Return live training progress as JSON.
    The browser polls this every second to update the UI.
    """
    return jsonify(TRAINING_STATUS)


@app.route("/results")
def results():
    """
    After training, return summary statistics as JSON.
    Called by the front-end once TRAINING_STATUS['done'] == True.
    """
    if not SUCCESS_HISTORY:
        return jsonify({"error": "No training data yet."}), 404

    total_ep    = len(SUCCESS_HISTORY)
    success_cnt = int(sum(SUCCESS_HISTORY))
    last_100    = SUCCESS_HISTORY[-100:] if len(SUCCESS_HISTORY) >= 100 else SUCCESS_HISTORY
    win_rate    = round(sum(last_100) / len(last_100) * 100, 1)
    avg_reward  = round(float(np.mean(REWARDS_HISTORY)), 2)

    return jsonify({
        "total_episodes": total_ep,
        "successes":      success_cnt,
        "win_rate_last_100": win_rate,
        "avg_reward":     avg_reward,
    })


@app.route("/plots")
def plots():
    """
    Generate Matplotlib plots and return them as base64-encoded PNG
    strings inside a JSON object, so the front-end can embed them
    directly in <img src="data:image/png;base64,…"> tags.
    """
    if not REWARDS_HISTORY:
        return jsonify({"error": "No training data available yet."}), 404

    rewards  = np.array(REWARDS_HISTORY, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)

    # Rolling window for smoothing
    window = min(50, len(rewards))
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

    # ── Plot 1: Reward over episodes ──────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    fig1.patch.set_facecolor("#12172a")
    ax1.set_facecolor("#1a2035")

    ax1.plot(episodes, rewards, color="#2a4a7f", alpha=0.4, linewidth=0.8, label="Raw reward")
    ax1.plot(
        episodes[window - 1:], smooth,
        color="#4cc9f0", linewidth=2.2, label=f"Rolling avg (n={window})"
    )
    ax1.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Episode", color="#aab4c8")
    ax1.set_ylabel("Total Reward", color="#aab4c8")
    ax1.set_title("Reward per Episode", color="#e0e8ff", fontsize=13, pad=12)
    ax1.tick_params(colors="#aab4c8")
    ax1.spines[:].set_color("#2a3555")
    ax1.legend(facecolor="#1a2035", edgecolor="#2a3555", labelcolor="#aab4c8")
    plt.tight_layout()

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=120, facecolor=fig1.get_facecolor())
    buf1.seek(0)
    img1_b64 = base64.b64encode(buf1.read()).decode("utf-8")
    plt.close(fig1)

    # ── Plot 2: Rolling success rate ──────────────────────────────────
    successes = np.array(SUCCESS_HISTORY, dtype=float)
    win_smooth = np.convolve(successes, np.ones(window) / window, mode="valid") * 100

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    fig2.patch.set_facecolor("#12172a")
    ax2.set_facecolor("#1a2035")

    ax2.fill_between(
        episodes[window - 1:], win_smooth,
        alpha=0.25, color="#7bed9f"
    )
    ax2.plot(
        episodes[window - 1:], win_smooth,
        color="#7bed9f", linewidth=2.2, label=f"Success rate (rolling n={window})"
    )
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Episode", color="#aab4c8")
    ax2.set_ylabel("Success Rate", color="#aab4c8")
    ax2.set_title("Agent Success Rate over Training", color="#e0e8ff", fontsize=13, pad=12)
    ax2.tick_params(colors="#aab4c8")
    ax2.spines[:].set_color("#2a3555")
    ax2.legend(facecolor="#1a2035", edgecolor="#2a3555", labelcolor="#aab4c8")
    plt.tight_layout()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=120, facecolor=fig2.get_facecolor())
    buf2.seek(0)
    img2_b64 = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close(fig2)

    return jsonify({"reward_plot": img1_b64, "success_plot": img2_b64})


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Tactical Convoy Evasion – Flask Dashboard")
    print("  Open  http://127.0.0.1:5000  in your browser")
    print("=" * 55)
    # use_reloader=False is critical: the Pygame thread cannot survive
    # Flask's hot-reloader forking a second process.
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
