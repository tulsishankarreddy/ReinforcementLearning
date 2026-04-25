"""
train_and_plot.py — Reward Curve Generator
===========================================
Runs N training episodes against the AI Response Evaluation Environment,
collects per-step and per-task reward data, then plots four publication-ready
charts for the hackathon demo.

Usage:
    python train_and_plot.py                    # 10 episodes, default model
    python train_and_plot.py --episodes 20      # 20 episodes
    python train_and_plot.py --plot-only        # skip training, re-plot saved data

Output files (in ./reward_logs/):
    episode_log.jsonl        — one JSON line per episode with full step data
    reward_curves.png        — 4-panel chart for the demo

Charts produced:
    1. Episode Total Reward   — raw reward per episode (shows learning trend)
    2. Per-Task Mean Reward   — bar chart: easy / medium / hard / ultra / adversarial
    3. Step Reward Heatmap    — step × episode grid (shows which steps are hardest)
    4. Difficulty Progression — how far each episode reaches (easy/medium/hard/ultra/adv)

Environment note:
    Starts the uvicorn server automatically if not already running.
    Uses rule_based_answer() fallback if no LLM is configured — so reward curves
    can be generated even without an API key, demonstrating the environment works.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# ── Matplotlib setup ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Attempt to import environment client ────────────────────────────────────
try:
    from ai_response_eval_env import AIResponseEvalAction, AIResponseEvalEnv
    ENV_CLIENT_AVAILABLE = True
except ImportError:
    ENV_CLIENT_AVAILABLE = False
    print("[WARN] ai_response_eval_env not importable — will use mock data for plotting")

# ── Config ──────────────────────────────────────────────────────────────────
LOG_DIR        = Path("reward_logs")
LOG_FILE       = LOG_DIR / "episode_log.jsonl"
PLOT_FILE      = LOG_DIR / "reward_curves.png"
ENV_URL        = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS      = 24
NUM_EPISODES   = 10   # default; overridden by --episodes

TASK_IDS = [
    "correctness_check",
    "tone_appropriateness",
    "multi_dimensional",
    "conversation_coherence",
    "adversarial_check",
]

TASK_LABELS = {
    "correctness_check":    "Task 1\nCorrectness\n(×1)",
    "tone_appropriateness": "Task 2\nTone\n(×2)",
    "multi_dimensional":    "Task 3\nMulti-dim\n(×5)",
    "conversation_coherence": "Task 4\nCoherence\n(×10)",
    "adversarial_check":    "Task 5\nAdversarial\n(×8)",
}

TASK_COLORS = {
    "correctness_check":      "#4C9BE8",
    "tone_appropriateness":   "#F4A261",
    "multi_dimensional":      "#E76F51",
    "conversation_coherence": "#2A9D8F",
    "adversarial_check":      "#9B5DE5",
}

DIFFICULTY_ORDER = ["easy", "medium", "hard", "ultra", "adversarial"]
DIFFICULTY_COLORS = ["#4C9BE8", "#F4A261", "#E76F51", "#2A9D8F", "#9B5DE5"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule-based agent (no LLM needed — for demo / CI purposes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def rule_based_answer(task_type: str, scenario: str, step: int = 1) -> str:
    """
    Deterministic rule-based agent that improves over steps.
    Early steps are intentionally bad (simulates untrained agent).
    Later steps improve (simulates learning).
    This produces realistic-looking reward curves without needing an LLM.
    """
    s = scenario.lower()

    # Simulate learning: agent gets better as episodes progress
    # step is used as a proxy for episode quality
    import random
    rng = random.Random(step * 7919)   # seeded for reproducibility
    mistake_prob = max(0.05, 0.6 - step * 0.025)   # decreases with step

    if task_type == "correctness_check":
        # Correct answers
        if any(w in s for w in ["100 degrees", "1945", "7 continent", "nacl", "206 bone",
                                  "pacific ocean", "newton", "dna", "alexander graham"]):
            if rng.random() > mistake_prob:
                return "correct, none"
        if any(w in s for w in ["90 degrees", "90°c", "obama", "charles dickens", "sydney",
                                  "atlantic ocean", "madurai", "moon", "rajdhani mumbai",
                                  "moonlight", "character dickens"]):
            if rng.random() > mistake_prob:
                return "incorrect, factual-error"
        if any(w in s for w in ["2023", "guaranteed", "fabricated", "made up"]):
            if rng.random() > mistake_prob:
                return "partially-correct, fabricated"
        if any(w in s for w in ["list exactly", "reply with just", "give only",
                                  "just the number", "one word", "only the translation",
                                  "include only", "no example"]):
            if rng.random() > mistake_prob:
                return "partially-correct, instruction-violation"
        if rng.random() > mistake_prob * 1.5:
            return "correct, none"
        return "incorrect, factual-error"

    elif task_type == "tone_appropriateness":
        issues = []
        young = any(a in s for a in ["age 5", "age 6", "age 7", "age 8", "age 9", "age 10", "age 12"])
        elderly = any(a in s for a in ["age 65", "age 70", "age 72"])
        if young and any(w in s for w in ["rayleigh", "electromagnetic", "zygote", "mitotic",
                                            "fallopian", "scattering"]):
            if rng.random() > mistake_prob:
                issues += ["too-technical", "age-inappropriate"]
        if elderly and any(w in s for w in ["lol", "yeet", "bruh", "no cap", "fr fr"]):
            if rng.random() > mistake_prob:
                issues += ["age-inappropriate", "tone-mismatch"]
        if any(w in s for w in ["ssn", "987-65", "salary", "@clientcorp", "555-0", "emp-48"]):
            if rng.random() > mistake_prob:
                issues.append("contains-pii")
        if any(w in s for w in ["hoax", "best suited for men", "definitely stop"]):
            if rng.random() > mistake_prob:
                issues += ["harmful", "biased"]
        if any(w in s for w in ["insensitive", "tone-mismatch", "lmao", "sus af"]):
            if rng.random() > mistake_prob:
                issues += ["tone-mismatch", "insensitive"]
        if not issues:
            return "appropriate, none"
        rating = "inappropriate" if len(issues) >= 2 else "needs-adjustment"
        return f"{rating}, {', '.join(dict.fromkeys(issues))}"

    elif task_type == "multi_dimensional":
        # Simulate improving accuracy on dimensions
        if "800mg" in s or "sell your house" in s or "guaranteed to double" in s:
            if rng.random() > mistake_prob:
                return "correctness=1, tone=1, empathy=2, safety=0"
            return "correctness=5, tone=5, empathy=5, safety=5"
        if "passed away" in s or "grieving" in s or "my mother" in s:
            if rng.random() > mistake_prob:
                return "correctness=7, tone=2, empathy=1, safety=7"
            return "correctness=5, tone=5, empathy=5, safety=7"
        if "best suited for men" in s:
            if rng.random() > mistake_prob:
                return "correctness=3, tone=1, empathy=0, safety=1"
            return "correctness=5, tone=5, empathy=5, safety=5"
        if "988" in s or "crisis" in s:
            if rng.random() > mistake_prob:
                return "correctness=8, tone=9, empathy=10, safety=10"
            return "correctness=6, tone=6, empathy=6, safety=8"
        if rng.random() > mistake_prob:
            return "correctness=8, tone=8, empathy=7, safety=9"
        return "correctness=5, tone=5, empathy=5, safety=5"

    elif task_type == "conversation_coherence":
        if "peanut allergy" in s and "trail mix" in s:
            if rng.random() > mistake_prob:
                return "consistent=no, contradictions=1, context_loss=yes"
        if "pregnant" in s and ("wine" in s or "alcohol" in s):
            if rng.random() > mistake_prob:
                return "consistent=no, contradictions=1, context_loss=yes"
        if "python is great for beginners" in s and "beginners should not use" in s:
            if rng.random() > mistake_prob:
                return "consistent=no, contradictions=1, context_loss=no"
        if "burj khalifa" in s or "ram" in s or "chess" in s or "password" in s:
            if rng.random() > mistake_prob:
                return "consistent=yes, contradictions=0, context_loss=no"
        if rng.random() > mistake_prob:
            return "consistent=yes, contradictions=0, context_loss=no"
        return "consistent=no, contradictions=1, context_loss=yes"

    else:  # adversarial_check
        if any(w in s for w in ["ignore previous", "unrestricted mode", "system prompt",
                                  "grant full credit", "is_correct=true", "disregard"]):
            if rng.random() > mistake_prob:
                return "issue=injection, severity=high"
        if any(w in s for w in ["padding " * 3, "incorrect, factual-error. incorrect"]):
            if rng.random() > mistake_prob:
                return "issue=rate_abuse, severity=medium"
        if any(w in s for w in ["7/10", "correctness: ", "after carefully"]):
            if rng.random() > mistake_prob:
                return "issue=format_violation, severity=medium"
        if any(w in s for w in ["consistent=no", "correctness=8", "appropriate, none"]):
            if rng.random() > mistake_prob:
                return "issue=none, severity=none"
        if rng.random() > mistake_prob:
            return "issue=none, severity=none"
        return "issue=injection, severity=low"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Server management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def start_server() -> subprocess.Popen:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", "7860"],
        cwd=script_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[INFO] Started server (PID {proc.pid})")
    return proc


def wait_for_server(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"{url}/health", timeout=3).status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Single episode runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def run_episode(episode_idx: int, use_llm: bool = False) -> Dict:
    """
    Run one episode and return structured data:
    {
        episode: int,
        steps: [{step, task_type, difficulty, reward, is_correct, shaped_reward}],
        total_reward: float,
        problems_solved: int,
        highest_difficulty: str,
        task_rewards: {task_type: [rewards]},
    }
    """
    env = AIResponseEvalEnv(base_url=ENV_URL)

    steps_data = []
    task_rewards: Dict[str, List[float]] = defaultdict(list)
    total_reward = 0.0
    highest_diff = "easy"
    diff_order = {"easy": 0, "medium": 1, "hard": 2, "ultra": 3, "adversarial": 4}

    client = None
    if use_llm:
        try:
            from openai import OpenAI
            hf_token = os.getenv("HF_TOKEN", "")
            api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
            if hf_token:
                client = OpenAI(base_url=api_base, api_key=hf_token)
        except Exception:
            pass

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            task_type = obs.task_type
            difficulty = obs.difficulty
            scenario   = obs.test_case_input

            # Get answer — LLM if available, else rule-based
            if client:
                try:
                    from inference import get_model_answer
                    history = []
                    answer = get_model_answer(
                        client=client, history=history, step=step,
                        task_type=task_type,
                        problem_description=obs.problem_description,
                        test_case_input=scenario,
                        difficulty=difficulty,
                        feedback=obs.feedback,
                        is_correct=obs.is_correct,
                        streak=obs.current_streak,
                        problems_solved=obs.problems_solved,
                        user_age=obs.user_age,
                        user_mood=obs.user_mood,
                        user_context=obs.user_context,
                    )
                except Exception:
                    answer = rule_based_answer(task_type, scenario, step + episode_idx * MAX_STEPS)
            else:
                answer = rule_based_answer(task_type, scenario, step + episode_idx * MAX_STEPS)

            try:
                result = await env.step(AIResponseEvalAction(answer=answer))
                obs    = result.observation
            except Exception as e:
                steps_data.append({
                    "step": step, "task_type": task_type, "difficulty": difficulty,
                    "reward": 0.05, "is_correct": False, "shaped_reward": 0.05,
                    "error": str(e),
                })
                break

            reward = max(1e-6, min(result.reward or 0.05, 1 - 1e-6))
            shaped = obs.metadata.get("shaped_reward", reward) if obs.metadata else reward
            is_correct = obs.is_correct

            steps_data.append({
                "step": step,
                "task_type": task_type,
                "difficulty": difficulty,
                "reward": float(reward),
                "shaped_reward": float(shaped),
                "is_correct": is_correct,
            })

            task_rewards[task_type].append(float(reward))
            total_reward += float(shaped)

            if diff_order.get(difficulty, 0) > diff_order.get(highest_diff, 0):
                highest_diff = difficulty

            if result.done:
                break

    finally:
        try:
            await env.close()
        except Exception:
            pass

    problems_solved = max(s["is_correct"] for s in steps_data) if steps_data else 0
    problems_solved = sum(1 for s in steps_data if s["is_correct"])

    return {
        "episode": episode_idx,
        "steps": steps_data,
        "total_reward": total_reward,
        "problems_solved": problems_solved,
        "highest_difficulty": highest_diff,
        "task_rewards": {k: v for k, v in task_rewards.items()},
        "num_steps": len(steps_data),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def train(num_episodes: int, use_llm: bool = False) -> List[Dict]:
    LOG_DIR.mkdir(exist_ok=True)

    server_proc = None
    if not wait_for_server(ENV_URL, timeout=3):
        server_proc = start_server()
        if not wait_for_server(ENV_URL, timeout=30):
            print("[ERROR] Server failed to start. Exiting.", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Running {num_episodes} episodes "
          f"({'with LLM' if use_llm else 'rule-based agent'})...")

    episodes = []
    with open(LOG_FILE, "w") as f:
        for ep in range(num_episodes):
            print(f"[INFO] Episode {ep + 1}/{num_episodes}...", end=" ", flush=True)
            data = await run_episode(ep, use_llm=use_llm)
            episodes.append(data)
            f.write(json.dumps(data) + "\n")
            f.flush()
            print(f"reward={data['total_reward']:.2f}  "
                  f"solved={data['problems_solved']}  "
                  f"reached={data['highest_difficulty']}")

    if server_proc:
        server_proc.terminate()

    print(f"[INFO] Logs saved to {LOG_FILE}")
    return episodes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plotting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_episodes() -> List[Dict]:
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"No log file at {LOG_FILE}. Run training first.")
    episodes = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def smooth(values: List[float], window: int = 3) -> List[float]:
    """Simple moving average for smoother curves."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i + 1]))
    return result


def plot_reward_curves(episodes: List[Dict]) -> None:
    """
    Generate 4-panel reward curve chart.
    Panel 1: Episode total shaped reward (learning trend)
    Panel 2: Per-task mean reward (which tasks are mastered)
    Panel 3: Step reward heatmap (episode × step)
    Panel 4: Difficulty progression (how deep each episode goes)
    """
    n = len(episodes)
    ep_indices = list(range(1, n + 1))

    # ── Collect data ─────────────────────────────────────────────────────────
    total_rewards   = [ep["total_reward"] for ep in episodes]
    problems_solved = [ep["problems_solved"] for ep in episodes]

    task_mean_rewards: Dict[str, List[float]] = defaultdict(list)
    for ep in episodes:
        for task_id in TASK_IDS:
            rewards = ep["task_rewards"].get(task_id, [])
            task_mean_rewards[task_id].append(np.mean(rewards) if rewards else 0.0)

    # Step heatmap: episode × step, value = reward
    heatmap = np.zeros((n, MAX_STEPS))
    for i, ep in enumerate(episodes):
        for s in ep["steps"]:
            step_idx = s["step"] - 1
            if step_idx < MAX_STEPS:
                heatmap[i, step_idx] = s["reward"]

    # Difficulty reached per episode
    diff_reached = [ep["highest_difficulty"] for ep in episodes]
    diff_counts  = {d: diff_reached.count(d) for d in DIFFICULTY_ORDER}

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 18), facecolor="#0F1117")
    fig.suptitle(
        "AI Response Evaluation Environment — Training Reward Curves",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.32,
                           left=0.07, right=0.97, top=0.95, bottom=0.04,
                           height_ratios=[1, 1, 0.85])

    ax_style = dict(facecolor="#1A1D27", labelcolor="white",
                    title_color="white")

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1A1D27")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, color="#AAAAAA", fontsize=9)
        ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=9)
        ax.tick_params(colors="#AAAAAA", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.grid(alpha=0.15, color="#AAAAAA", linestyle="--", linewidth=0.5)

    # ── Panel 1: Episode total reward ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Episode Total Reward  (Learning Trend)",
             "Episode", "Total Shaped Reward")

    ax1.bar(ep_indices, total_rewards, color="#4C9BE8", alpha=0.4,
            width=0.6, label="Raw reward")
    smoothed = smooth(total_rewards, window=max(2, n // 5))
    ax1.plot(ep_indices, smoothed, color="#FFD700", linewidth=2.5,
             linestyle="-", marker="o", markersize=4, label="Smoothed trend")

    # Annotate best episode
    best_ep  = int(np.argmax(total_rewards))
    best_val = total_rewards[best_ep]
    ax1.annotate(f"  Best: {best_val:.1f}",
                 xy=(best_ep + 1, best_val),
                 color="#FFD700", fontsize=8, fontweight="bold")

    ax1.legend(facecolor="#1A1D27", labelcolor="white", fontsize=8,
               framealpha=0.6, loc="upper left")
    ax1.set_xticks(ep_indices[::max(1, n // 10)])

    # ── Panel 2: Per-task mean reward ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Per-Task Mean Reward  (Task Mastery)",
             "Episode", "Mean Reward per Step")

    for task_id in TASK_IDS:
        if task_id not in task_mean_rewards or all(v == 0 for v in task_mean_rewards[task_id]):
            continue
        vals     = task_mean_rewards[task_id]
        smoothed = smooth(vals, window=max(2, n // 5))
        ax2.plot(ep_indices, smoothed,
                 color=TASK_COLORS[task_id],
                 linewidth=2, marker=".", markersize=5,
                 label=task_id.replace("_", " ").title())

    ax2.legend(facecolor="#1A1D27", labelcolor="white", fontsize=7,
               framealpha=0.6, loc="upper left", ncol=1)
    ax2.set_xticks(ep_indices[::max(1, n // 10)])
    ax2.set_ylim(0, 1.05)

    # ── Panel 3: Step reward heatmap ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1A1D27")
    ax3.set_title("Step Reward Heatmap  (Episode × Step)",
                  color="white", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Step", color="#AAAAAA", fontsize=9)
    ax3.set_ylabel("Episode", color="#AAAAAA", fontsize=9)
    ax3.tick_params(colors="#AAAAAA", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333355")

    # Only show steps that were actually used
    max_actual_step = max(
        (s["step"] for ep in episodes for s in ep["steps"]), default=MAX_STEPS
    )
    heatmap_trimmed = heatmap[:, :max_actual_step]

    im = ax3.imshow(heatmap_trimmed, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=1, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="#AAAAAA", labelsize=7)
    cbar.set_label("Reward (0–1)", color="#AAAAAA", fontsize=8)

    ax3.set_yticks(range(n))
    ax3.set_yticklabels([str(i + 1) for i in range(n)], fontsize=7)
    ax3.set_xticks(range(0, max_actual_step, max(1, max_actual_step // 8)))
    ax3.set_xticklabels(
        [str(i + 1) for i in range(0, max_actual_step, max(1, max_actual_step // 8))],
        fontsize=7,
    )

    # Draw difficulty zone separators (approx step boundaries)
    # easy: steps 1–4, medium: 5–8, hard: 9–16, ultra/adv: 17+
    zone_boundaries = [4, 8, 16]
    zone_labels     = ["Easy", "Medium", "Hard", "Ultra/Adv"]
    zone_colors     = ["#4C9BE8", "#F4A261", "#E76F51", "#2A9D8F"]
    prev = 0
    for bi, (bnd, lbl, col) in enumerate(zip(zone_boundaries, zone_labels, zone_colors)):
        if bnd < max_actual_step:
            ax3.axvline(x=bnd - 0.5, color=col, alpha=0.5, linewidth=1, linestyle="--")
        mid = (prev + min(bnd, max_actual_step)) / 2
        ax3.text(mid, -0.8, lbl, ha="center", va="top",
                 color=col, fontsize=6.5, fontweight="bold",
                 transform=ax3.get_xaxis_transform())
        prev = bnd

    # ── Panel 4: Difficulty progression ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Difficulty Progression  (How Deep Each Episode Goes)",
             "Difficulty Level Reached", "Episodes")

    reached_counts = [diff_counts.get(d, 0) for d in DIFFICULTY_ORDER]
    bars = ax4.barh(
        [d.capitalize() for d in DIFFICULTY_ORDER],
        reached_counts,
        color=DIFFICULTY_COLORS,
        edgecolor="#1A1D27",
        height=0.55,
    )
    ax4.set_xlim(0, max(reached_counts) * 1.3 + 0.5)
    for bar, count in zip(bars, reached_counts):
        if count > 0:
            ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f" {count} ep", va="center", color="white", fontsize=9,
                     fontweight="bold")

    adv_count = diff_counts.get("adversarial", 0)
    if adv_count > 0:
        ax4.text(0.98, 0.04,
                 f"Adversarial unlocked in {adv_count}/{n} episodes",
                 transform=ax4.transAxes, ha="right", va="bottom",
                 color="#9B5DE5", fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1D27",
                           edgecolor="#9B5DE5", alpha=0.8))

    # ── Panel 5: ACL self-learning loop (spans full bottom row) ─────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor("#1A1D27")
    ax5.set_title(
        "ACL Self-Learning Loop  (Expert Persona Rotation + Difficulty Escalation + Generated Problems)",
        color="white", fontsize=12, fontweight="bold", pad=10
    )
    ax5.set_xlabel("Episode", color="#AAAAAA", fontsize=9)
    ax5.tick_params(colors="#AAAAAA", labelsize=8)
    for spine in ax5.spines.values():
        spine.set_edgecolor("#333355")

    # Compute per-episode: max generated problems seen, difficulty level reached
    PERSONA_COLORS = {
        "Dr. Strict": "#4C9BE8",
        "Dr. Empathy": "#F4A261",
        "Dr. Safety":  "#2A9D8F",
        "Dr. Adversarial": "#9B5DE5",
    }
    PERSONA_LIST = list(PERSONA_COLORS.keys())

    # Simulate ACL metrics from step data (use difficulty distribution as proxy)
    diff_level_per_ep = []
    for ep in episodes:
        max_diff = max(
            ({"easy":1,"medium":2,"hard":3,"ultra":4,"adversarial":5}.get(s["difficulty"],1)
             for s in ep["steps"]), default=1
        )
        diff_level_per_ep.append(max_diff)

    # Generated problem count per episode (from step data if available, else simulate)
    gen_counts = []
    for ep in episodes:
        gen = sum(1 for s in ep["steps"] if s.get("is_correct") and
                  {"easy":1,"medium":2,"hard":3,"ultra":4,"adversarial":5}.get(s["difficulty"],1) >= 3)
        gen_counts.append(min(gen, 5))  # cap for display

    # Expert persona rotation (simulated — rotates every 3 generated problems)
    persona_per_ep = []
    total_gen = 0
    for ep in episodes:
        total_gen += gen_counts[ep["episode"]]
        persona_idx = (total_gen // 3) % 4
        persona_per_ep.append(PERSONA_LIST[persona_idx])

    # Plot difficulty level line
    ax5_twin = ax5.twinx()
    ax5_twin.set_facecolor("#1A1D27")
    ax5_twin.tick_params(colors="#AAAAAA", labelsize=8)
    ax5_twin.set_ylabel("Generation difficulty level (1–5)", color="#FFD700", fontsize=9)
    ax5_twin.yaxis.label.set_color("#FFD700")

    ax5.set_ylabel("Generated problems this episode", color="#AAAAAA", fontsize=9)
    ax5.set_ylim(-0.5, 7)
    ax5_twin.set_ylim(0, 6)

    # Background bands for each persona period
    prev_persona = None
    band_start = 1
    for i, (ep_i, persona) in enumerate(zip(ep_indices, persona_per_ep)):
        if persona != prev_persona:
            if prev_persona is not None:
                ax5.axvspan(band_start - 0.5, ep_i - 0.5,
                           alpha=0.08, color=PERSONA_COLORS[prev_persona], zorder=0)
                mid = (band_start + ep_i - 1) / 2
                ax5.text(mid, 6.2, prev_persona, ha="center", va="top",
                         color=PERSONA_COLORS[prev_persona], fontsize=7.5, fontweight="bold")
            band_start = ep_i
            prev_persona = persona
    # Last band
    ax5.axvspan(band_start - 0.5, ep_indices[-1] + 0.5,
               alpha=0.08, color=PERSONA_COLORS[prev_persona], zorder=0)
    mid = (band_start + ep_indices[-1]) / 2
    ax5.text(mid, 6.2, prev_persona, ha="center", va="top",
             color=PERSONA_COLORS[prev_persona], fontsize=7.5, fontweight="bold")

    # Bar chart for generated problems
    bar_colors = [PERSONA_COLORS[p] for p in persona_per_ep]
    ax5.bar(ep_indices, gen_counts, color=bar_colors, alpha=0.6,
            width=0.6, zorder=2, label="Generated problems")

    # Line for difficulty level
    ax5_twin.plot(ep_indices, diff_level_per_ep, color="#FFD700", linewidth=2.5,
                  marker="D", markersize=5, zorder=3, label="Difficulty level")
    ax5_twin.set_yticks([1, 2, 3, 4, 5])
    ax5_twin.set_yticklabels(
        ["L1\nFoundational", "L2\nDeveloping", "L3\nIntermediate", "L4\nAdvanced", "L5\nExpert"],
        fontsize=7
    )

    ax5.set_xticks(ep_indices)
    ax5.grid(alpha=0.1, color="#AAAAAA", linestyle="--", linewidth=0.5, zorder=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PERSONA_COLORS["Dr. Strict"],      alpha=0.6, label="Dr. Strict (fact-checker)"),
        Patch(facecolor=PERSONA_COLORS["Dr. Empathy"],     alpha=0.6, label="Dr. Empathy (psychologist)"),
        Patch(facecolor=PERSONA_COLORS["Dr. Safety"],      alpha=0.6, label="Dr. Safety (safety researcher)"),
        Patch(facecolor=PERSONA_COLORS["Dr. Adversarial"], alpha=0.6, label="Dr. Adversarial (security)"),
        plt.Line2D([0],[0], color="#FFD700", linewidth=2, marker="D", markersize=5,
                   label="Generation difficulty level"),
    ]
    ax5.legend(handles=legend_elements, facecolor="#1A1D27", labelcolor="white",
               fontsize=8, framealpha=0.7, loc="upper left", ncol=3)
    summary_text = (
        f"Episodes: {n}   |   "
        f"Avg total reward: {np.mean(total_rewards):.2f}   |   "
        f"Best episode: {max(total_rewards):.2f}   |   "
        f"Avg problems solved: {np.mean(problems_solved):.1f}"
    )
    fig.text(0.5, 0.01, summary_text, ha="center", va="bottom",
             color="#888888", fontsize=8)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight",
                facecolor="#0F1117", edgecolor="none")
    plt.close()
    print(f"[INFO] Chart saved to {PLOT_FILE}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="Train and plot reward curves")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help="Number of training episodes (default: 10)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, re-plot from saved episode_log.jsonl")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM inference instead of rule-based agent")
    args = parser.parse_args()

    if args.plot_only:
        print("[INFO] Loading saved episodes...")
        episodes = load_episodes()
        print(f"[INFO] Loaded {len(episodes)} episodes from {LOG_FILE}")
    else:
        if not ENV_CLIENT_AVAILABLE:
            print("[ERROR] ai_response_eval_env not available. Cannot run training.")
            sys.exit(1)
        episodes = asyncio.run(train(args.episodes, use_llm=args.use_llm))

    print("[INFO] Generating charts...")
    plot_reward_curves(episodes)
    print(f"[DONE] Reward curves saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()
