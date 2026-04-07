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
    from ..models import CodeAssessmentAction, CodeAssessmentObservation
    from .code_assessment_environment import CodeAssessmentEnvironment, TASK_TYPES, TASK_INSTRUCTIONS, PROBLEMS
except (ImportError, ModuleNotFoundError):
    from models import CodeAssessmentAction, CodeAssessmentObservation
    from server.code_assessment_environment import CodeAssessmentEnvironment, TASK_TYPES, TASK_INSTRUCTIONS, PROBLEMS


# Create the base app with OpenEnv endpoints
app = create_app(
    CodeAssessmentEnvironment,
    CodeAssessmentAction,
    CodeAssessmentObservation,
    env_name="code_assessment_env",
    max_concurrent_envs=10,
)


# ─── Task enumeration endpoint ──────────────────────────────────────────────
@app.get("/tasks")
async def list_tasks():
    """Enumerate all tasks with their grader info and action schema."""
    tasks = []
    for difficulty, task_type in TASK_TYPES.items():
        tasks.append({
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

    env = CodeAssessmentEnvironment()
    env._difficulty = difficulty
    is_correct, score, feedback = env._grade(task_id, answer, problem)

    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "score": score,
        "is_correct": is_correct,
        "feedback": feedback,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
