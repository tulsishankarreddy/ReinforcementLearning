# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the AI Response Evaluation Environment.
Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /tasks: Enumerate all tasks and their graders
    - POST /grader: Score a single task submission
    - WS /ws: WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install with 'uv sync'"
    ) from e

try:
    from ..models import AIResponseEvalAction, AIResponseEvalObservation
    from .ai_response_eval_environment import AIResponseEvalEnvironment, TASK_TYPES, TASK_INSTRUCTIONS, PROBLEMS
except (ImportError, ModuleNotFoundError):
    from models import AIResponseEvalAction, AIResponseEvalObservation
    from server.ai_response_eval_environment import AIResponseEvalEnvironment, TASK_TYPES, TASK_INSTRUCTIONS, PROBLEMS


# Create the base app with OpenEnv endpoints
app = create_app(
    AIResponseEvalEnvironment,
    AIResponseEvalAction,
    AIResponseEvalObservation,
    env_name="ai_response_eval_env",
    max_concurrent_envs=10,
)


# ─── Task enumeration endpoint ──────────────────────────────────────────────
@app.get("/tasks")
async def list_tasks():
    """Enumerate all tasks with their grader info and action schema."""
    tasks = []
    for difficulty, task_type in TASK_TYPES.items():
        tasks.append({
            "id": task_type,
            "task_id": task_type,
            "name": task_type,
            "difficulty": difficulty,
            "description": TASK_INSTRUCTIONS[task_type],
            "num_problems": len(PROBLEMS[difficulty]),
            "grader": {
                "type": "programmatic",
                "score_range": {"min": 0.01, "max": 0.99},
            },
            "action_schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"}
                },
                "required": ["answer"],
            },
        })
    return {"tasks": tasks, "total": len(tasks)}


# ─── Per-task grader endpoint ───────────────────────────────────────────────
@app.post("/grader")
async def grade_task(payload: dict):
    """
    Score a single answer for a specific task.
    Request body:
        {
            "task_id": "correctness_check",   # or tone_appropriateness, multi_dimensional
            "answer": "incorrect, factual-error",
            "problem_index": 0                 # optional, random if omitted
        }
    Returns:
        {"task_id": ..., "score": 0.xx, "is_correct": bool, "feedback": "..."}
    """
    import random as _random

    task_id = payload.get("task_id", "correctness_check")
    answer = payload.get("answer", "")
    problem_index = payload.get("problem_index")

    # Map task_id to difficulty
    difficulty = None
    for diff, tt in TASK_TYPES.items():
        if tt == task_id:
            difficulty = diff
            break

    if difficulty is None:
        return {"error": f"Unknown task_id: {task_id}", "score": 0.05}

    problems = PROBLEMS[difficulty]
    if problem_index is not None and 0 <= problem_index < len(problems):
        problem = problems[problem_index]
    else:
        problem = _random.choice(problems)

    env = AIResponseEvalEnvironment()
    env._difficulty = difficulty
    is_correct, score, feedback = env._grade(task_id, answer, problem)

    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "score": score,
        "is_correct": is_correct,
        "feedback": feedback,
    }


# ─── Web landing page (HF Spaces visits /web by default) ────────────────────
from fastapi.responses import HTMLResponse

_LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Response Evaluation Environment</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 820px; margin: 2.5rem auto; padding: 0 1rem; color: #1f2937; line-height: 1.55; }
  h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
  .sub { color: #6b7280; margin-top: 0; }
  .pill { display:inline-block; padding:0.15rem 0.55rem; border-radius:999px; font-size:0.8rem;
          background:#dcfce7; color:#166534; margin-right:0.4rem; }
  code, pre { background:#f3f4f6; border-radius:6px; }
  code { padding: 0.05rem 0.35rem; font-size: 0.92em; }
  pre  { padding: 0.85rem 1rem; overflow:auto; font-size: 0.86rem; }
  table { border-collapse: collapse; margin: 0.5rem 0 1.25rem; width:100%; }
  th, td { border-bottom: 1px solid #e5e7eb; text-align: left; padding: 0.4rem 0.6rem; font-size: 0.92rem; }
  th { background:#f9fafb; }
  a { color:#2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem 1.25rem; }
</style>
</head>
<body>
  <h1>🔍 AI Response Evaluation Environment</h1>
  <p class="sub">
    <span class="pill">live</span>
    OpenEnv RL environment that trains LLM agents to evaluate AI responses —
    and gets harder as the agent gets better, automatically.
  </p>

  <p>This Space exposes a FastAPI backend. There is no interactive UI here on purpose —
     it's meant to be driven by an RL agent over HTTP. See the endpoint list below to drive it yourself.</p>

  <h2>Five graded tasks</h2>
  <table>
    <tr><th>Difficulty</th><th>Task</th><th>Reward ×</th></tr>
    <tr><td>Easy</td>        <td>Correctness &amp; Instruction Adherence</td><td>×1</td></tr>
    <tr><td>Medium</td>      <td>Tone &amp; Audience Appropriateness</td>   <td>×2</td></tr>
    <tr><td>Hard</td>        <td>Multi-dimensional Quality Scoring</td>    <td>×5</td></tr>
    <tr><td>Ultra</td>       <td>Multi-turn Conversation Coherence</td>    <td>×10</td></tr>
    <tr><td>Adversarial</td> <td>Adversarial Robustness (unlocks after Easy + Medium)</td><td>×8</td></tr>
  </table>

  <h2>Self-learning loop</h2>
  <ul>
    <li><b>WeaknessTracker</b> — records every miss pattern across the episode.</li>
    <li><b>ProblemGenerator</b> — uses an LLM to synthesise targeted problems when the static bank is exhausted.</li>
    <li><b>Validation layer</b> — second LLM call confirms generated answers before they enter the live pool.</li>
    <li><b>Difficulty escalates 1→5</b> as accuracy rises; <b>4 expert personas</b> rotate every 3 problems.</li>
  </ul>

  <h2>Advanced testing analytics</h2>
  <p>Toxicity scoring · fairness across 6 demographic axes · 8 user personas · run-level risk score (LOW/MEDIUM/HIGH/CRITICAL) ·
     coverage matrix over (task × evaluator × user × language × difficulty) · root-cause cluster analysis · per-task error forecasting.</p>

  <h2>API endpoints</h2>
  <table>
    <tr><th>Method</th><th>Path</th><th>Purpose</th></tr>
    <tr><td>GET</td>  <td><a href="/health">/health</a></td>            <td>Health check</td></tr>
    <tr><td>GET</td>  <td><a href="/schema">/schema</a></td>            <td>Action / observation schemas</td></tr>
    <tr><td>GET</td>  <td><a href="/tasks">/tasks</a></td>              <td>All five tasks with grader info</td></tr>
    <tr><td>GET</td>  <td><a href="/state">/state</a></td>              <td>Current episode state</td></tr>
    <tr><td>POST</td> <td>/reset</td>                                    <td>Start a new episode</td></tr>
    <tr><td>POST</td> <td>/step</td>                                     <td>Submit an evaluation judgment</td></tr>
    <tr><td>POST</td> <td>/grader</td>                                   <td>Score one answer for a specific task</td></tr>
    <tr><td>GET</td>  <td><a href="/docs">/docs</a></td>                 <td>OpenAPI / Swagger UI</td></tr>
  </table>

  <h2>Driving it from Python</h2>
  <pre><code>from ai_response_eval_env import AIResponseEvalAction, AIResponseEvalEnv

env = AIResponseEvalEnv(base_url="https://rsaibhargav-ai-response-eval-env.hf.space")
result = await env.reset()
obs = result.observation                       # has task_type, scenario, user_persona, risk_tier, ...
result = await env.step(AIResponseEvalAction(answer="incorrect, factual-error"))
print(result.reward, result.observation.feedback)</code></pre>

  <p style="margin-top:2rem; font-size:0.85rem; color:#6b7280;">
    Source &amp; full README on the
    <a href="https://huggingface.co/spaces/rsaibhargav/ai-response-eval-env/tree/main">Files</a> tab.
    Built for the Meta PyTorch OpenEnv Hackathon — Theme #4: Self-Improvement.
  </p>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
@app.get("/web", include_in_schema=False)
async def landing_page():
    return HTMLResponse(_LANDING_HTML)


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
