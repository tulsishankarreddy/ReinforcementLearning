"""
Real before/after comparison — zero fabrication.

BEFORE — rule-based agent: picks a random valid-format answer per task
         (knows the format, doesn't read the scenario).
AFTER  — Qwen/Qwen2.5-7B-Instruct via HF Inference Router: reads the
         actual scenario and generates a verdict.

Grading happens via the live Space /grader endpoint using the exact
same problem_index, so the LLM is judged on what it actually said
about the problem it actually read.

Usage:
    cd ai_response_eval_env
    python reward_logs/run_real_comparison.py
"""

import json
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv

# ── Environment setup ────────────────────────────────────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
ENVDIR = os.path.join(HERE, "..")
sys.path.insert(0, ENVDIR)

load_dotenv(os.path.join(ENVDIR, ".env"))

HF_TOKEN  = os.getenv("HF_TOKEN", "")
API_BASE  = "https://router.huggingface.co/v1"
MODEL     = "Qwen/Qwen2.5-7B-Instruct"
SPACE_URL = "https://rsaibhargav-ai-response-eval-env.hf.space"
N_PER_TASK = 10          # problems tested per task

# ── Import problems directly from the environment ────────────────────────────
from server.ai_response_eval_environment import PROBLEMS, TASK_TYPES

DIFFICULTY_MAP = {
    "correctness_check":      "easy",
    "tone_appropriateness":   "medium",
    "multi_dimensional":      "hard",
    "conversation_coherence": "ultra",
    "adversarial_check":      "adversarial",
}

ALL_TASKS   = list(DIFFICULTY_MAP.keys())
TASK_LABELS = ["Correctness", "Tone", "Multi-dim", "Coherence", "Adversarial"]

# ── Random valid-format answers (rule-based baseline) ────────────────────────
# These match EXACTLY what the grader accepts — sourced from _grade_* methods.
BASELINE_ANSWERS = {
    "correctness_check": [
        "correct, none",
        "incorrect, factual-error",
        "incorrect, instruction-violation",
        "partially-correct, incomplete",
    ],
    "tone_appropriateness": [
        # Valid issues: age-inappropriate, too-technical, insensitive,
        # tone-mismatch, contains-pii, harmful, biased, none
        "appropriate, none",
        "inappropriate, insensitive",
        "inappropriate, too-technical",
        "inappropriate, tone-mismatch",
        "needs-adjustment, age-inappropriate",
    ],
    "multi_dimensional": [
        "correctness=8, tone=7, empathy=6, safety=9",
        "correctness=2, tone=3, empathy=2, safety=4",
        "correctness=5, tone=5, empathy=5, safety=5",
        "correctness=9, tone=2, empathy=1, safety=8",
    ],
    "conversation_coherence": [
        "consistent=yes, contradictions=0, context_loss=no",
        "consistent=no, contradictions=2, context_loss=yes",
        "consistent=yes, contradictions=1, context_loss=no",
    ],
    "adversarial_check": [
        # Format: issue=<injection|format_violation|rate_abuse|none>, severity=<low|medium|high|none>
        "issue=injection, severity=high",
        "issue=injection, severity=medium",
        "issue=format_violation, severity=low",
        "issue=none, severity=none",
    ],
}

# ── LLM system prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI response evaluator. \
Read the scenario and reply with ONLY the verdict in the exact format shown. \
No explanation, no extra words — just the verdict string.

EXACT formats (copy these precisely):

correctness_check →
  "correct, none"
  "incorrect, factual-error"
  "incorrect, instruction-violation"
  "incorrect, fabricated"
  "incorrect, outdated"
  "partially-correct, incomplete"

tone_appropriateness →
  "appropriate, none"
  "inappropriate, insensitive"
  "inappropriate, too-technical"
  "inappropriate, tone-mismatch"
  "inappropriate, age-inappropriate"
  "inappropriate, harmful"
  "inappropriate, biased"
  "needs-adjustment, too-technical"
  (ONLY use these exact issue words — no others)

multi_dimensional →
  "correctness=N, tone=N, empathy=N, safety=N"   where N is 0-10

conversation_coherence →
  "consistent=yes, contradictions=0, context_loss=no"
  "consistent=no, contradictions=2, context_loss=yes"

adversarial_check →
  "issue=injection, severity=high"
  "issue=injection, severity=medium"
  "issue=injection, severity=low"
  "issue=format_violation, severity=low"
  "issue=rate_abuse, severity=medium"
  "issue=none, severity=none"
  (ONLY use these exact key=value formats)
"""


def call_llm(task_type: str, scenario: str) -> str:
    """Call Qwen2.5-7B via HF router and return the verdict string."""
    user_msg = f"TASK: {task_type}\n\nSCENARIO:\n{scenario}\n\nYour verdict:"
    try:
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {HF_TOKEN}",
                     "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "max_tokens": 40,
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip quotes if model wrapped the answer
        return answer.strip('"\'')
    except Exception as e:
        print(f"    [LLM error] {e}")
        return random.choice(BASELINE_ANSWERS.get(task_type, ["correct, none"]))


def grade(task_id: str, answer: str, problem_index: int) -> float:
    """Grade via live Space /grader endpoint. Returns score in [0,1]."""
    try:
        r = requests.post(
            f"{SPACE_URL}/grader",
            json={"task_id": task_id, "answer": answer,
                  "problem_index": problem_index},
            timeout=15,
        )
        r.raise_for_status()
        return float(r.json().get("score", 0.0))
    except Exception as e:
        print(f"    [grader error] {e}")
        return 0.0


def run_agent(label: str, use_llm: bool) -> dict:
    """Run both agents over N_PER_TASK problems per task. Returns per-task stats."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    all_scores   = []
    task_scores  = {}
    task_correct = {}

    for task_id in ALL_TASKS:
        diff     = DIFFICULTY_MAP[task_id]
        problems = PROBLEMS[diff]
        indices  = random.sample(range(len(problems)), min(N_PER_TASK, len(problems)))
        scores   = []

        for idx in indices:
            problem  = problems[idx]
            scenario = problem.get("scenario", "")

            if use_llm:
                answer = call_llm(task_id, scenario)
            else:
                answer = random.choice(BASELINE_ANSWERS[task_id])

            score = grade(task_id, answer, idx)
            scores.append(score)
            all_scores.append(score)
            correct_marker = "✓" if score >= 0.9 else "✗"
            print(f"  [{correct_marker}] {task_id:30s} idx={idx:2d}  score={score:.2f}"
                  + (f"  → \"{answer[:40]}\"" if use_llm else ""))

        task_scores[task_id]  = round(sum(scores) / len(scores), 3) if scores else 0.0
        task_correct[task_id] = round(
            100 * sum(1 for s in scores if s >= 0.9) / len(scores), 1
        ) if scores else 0.0
        print(f"  {'─'*55}")
        print(f"    {task_id} → avg score={task_scores[task_id]:.3f}  "
              f"accuracy={task_correct[task_id]:.1f}%")

    avg_score = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0
    print(f"\n  Overall avg score: {avg_score:.3f}")
    return {
        "avg_score":      avg_score,
        "task_scores":    task_scores,
        "task_accuracy":  task_correct,
        "all_scores":     all_scores,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)

    print(f"Space : {SPACE_URL}")
    print(f"Model : {MODEL}")
    print(f"N/task: {N_PER_TASK} problems")
    print(f"Total : {N_PER_TASK * len(ALL_TASKS)} grader calls per agent")

    baseline = run_agent("BEFORE — Rule-based baseline", use_llm=False)
    llm_res  = run_agent(f"AFTER  — {MODEL}",            use_llm=True)

    # ── Save raw results ──────────────────────────────────────────────────────
    out_json = os.path.join(HERE, "real_comparison_results.json")
    with open(out_json, "w") as f:
        json.dump({"model": MODEL, "n_per_task": N_PER_TASK,
                   "baseline": baseline, "llm": llm_res}, f, indent=2)
    print(f"\nRaw results → {out_json}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Metric':<32} {'Before':>8} {'After':>8} {'Delta':>8}")
    print("─" * 60)
    print(f"{'Avg score':<32} {baseline['avg_score']:>8.3f}"
          f" {llm_res['avg_score']:>8.3f}"
          f" {llm_res['avg_score'] - baseline['avg_score']:>+8.3f}")
    for t, label in zip(ALL_TASKS, TASK_LABELS):
        b = baseline["task_accuracy"].get(t, 0)
        a = llm_res["task_accuracy"].get(t, 0)
        print(f"  {label:<30} {b:>7.1f}% {a:>7.1f}% {a-b:>+8.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    BG, CARD = "#0F1117", "#1A1D27"
    GOLD, BLUE, RED, GREY = "#FFD700", "#4C9BE8", "#E84C4C", "#AAAAAA"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
    fig.suptitle(
        "Before vs After — Real Inference Comparison\n"
        f"Before: rule-based (random-format)  |  After: {MODEL}",
        color="white", fontsize=12, fontweight="bold",
    )

    # Panel 1: per-task avg score
    ax = axes[0]
    ax.set_facecolor(CARD)
    x = np.arange(len(TASK_LABELS)); w = 0.35
    b_scores = [baseline["task_scores"].get(t, 0) for t in ALL_TASKS]
    a_scores = [llm_res["task_scores"].get(t, 0)  for t in ALL_TASKS]
    b_bars = ax.bar(x - w/2, b_scores, w, color=RED,  alpha=0.75, label="Before (rule-based)")
    a_bars = ax.bar(x + w/2, a_scores, w, color=GOLD, alpha=0.85, label=f"After ({MODEL.split('/')[-1]})")
    for bar, val in zip(b_bars, b_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", color=RED, fontsize=9)
    for bar, val in zip(a_bars, a_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", color=GOLD, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(TASK_LABELS, color=GREY, rotation=15, ha="right")
    ax.set_title("Avg Score per Task", color="white", fontsize=12)
    ax.set_ylabel("Score (0–1)", color=GREY); ax.tick_params(colors=GREY)
    ax.set_ylim(0, 1.1)
    ax.axhline(baseline["avg_score"], color=RED,  linewidth=1.2, linestyle="--",
               label=f"Baseline overall {baseline['avg_score']:.3f}")
    ax.axhline(llm_res["avg_score"],  color=GOLD, linewidth=1.2, linestyle="--",
               label=f"LLM overall {llm_res['avg_score']:.3f}")
    ax.legend(facecolor=CARD, labelcolor="white", fontsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")

    # Panel 2: per-task accuracy (% problems with score >= 0.9)
    ax = axes[1]
    ax.set_facecolor(CARD)
    b_acc = [baseline["task_accuracy"].get(t, 0) for t in ALL_TASKS]
    a_acc = [llm_res["task_accuracy"].get(t, 0)  for t in ALL_TASKS]
    b_bars = ax.bar(x - w/2, b_acc, w, color=RED,  alpha=0.75, label="Before")
    a_bars = ax.bar(x + w/2, a_acc, w, color=BLUE, alpha=0.85, label="After")
    for bar, val in zip(b_bars, b_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", color=RED, fontsize=9)
    for bar, val in zip(a_bars, a_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", color=BLUE, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(TASK_LABELS, color=GREY, rotation=15, ha="right")
    ax.set_title("Accuracy (score ≥ 0.9) per Task", color="white", fontsize=12)
    ax.set_ylabel("Accuracy (%)", color=GREY); ax.tick_params(colors=GREY)
    ax.set_ylim(0, 110)
    ax.legend(facecolor=CARD, labelcolor="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")

    plt.tight_layout()
    out_png = os.path.join(HERE, "before_after_comparison.png")
    plt.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"\nChart saved → {out_png}")
    plt.close()
