"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

from code_assessment_env import CodeAssessmentAction, CodeAssessmentEnv
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "code_output_assessment")
BENCHMARK = os.getenv("BENCHMARK", "first_rl_proj")
MAX_STEPS = 15
TEMPERATURE = 0.7
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

# Max possible reward with normalized grading (0-1) × difficulty multipliers:
# Easy (1x): ~5 problems × 1.0 = 5.0
# Medium (2x): ~5 problems × 2.0 = 10.0
# Hard (5x): ~5 problems × 5.0 = 25.0
# Streak bonuses: ~3-4 bonuses × 0.5 = 1.5-2.0
# Total possible: ~40.0 with perfect performance
MAX_TOTAL_REWARD = 40.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are solving coding problems at different difficulty levels.
    
    For each problem:
    1. Read the problem description carefully
    2. Look at the test case input provided
    3. Calculate or determine the correct output
    4. Respond with ONLY the answer - no explanations, just the exact output value
    
    Examples:
    - If asked to add "3,5", respond: 8
    - If asked to reverse "hello", respond: olleh  
    - If asked for palindrome check "racecar", respond: true
    
    Be precise with formatting:
    - For lists, use comma-separated values: "1,2,3"
    - For true/false, use lowercase: "true" or "false"
    - For numbers, no extra spaces or characters
    
    You'll get higher rewards for:
    - Correct answers (especially on hard problems)
    - Maintaining a streak of correct answers
    - Solving problems quickly
    
    Focus on accuracy. Partial credit is available for close answers.  
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(
    step: int, 
    problem: str, 
    test_input: str, 
    difficulty: str,
    feedback: str,
    is_correct: bool,
    streak: int,
    problems_solved: int
) -> str:
    status = "✓ CORRECT!" if is_correct else feedback
    
    return textwrap.dedent(
        f"""
        Step {step}/15 | Difficulty: {difficulty.upper()} | Solved: {problems_solved} | Streak: {streak}
        
        Problem: {problem}
        Test Input: {test_input}
        
        Previous Feedback: {status}
        
        What is the output? (respond with just the answer)
        """
    ).strip()


def get_model_answer(
    client: OpenAI,
    step: int,
    problem: str,
    test_input: str,
    difficulty: str,
    feedback: str,
    is_correct: bool,
    streak: int,
    problems_solved: int
) -> str:
    user_prompt = build_user_prompt(step, problem, test_input, difficulty, feedback, is_correct, streak, problems_solved)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "0"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "0"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await CodeAssessmentEnv.from_docker_image(LOCAL_IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Get model's answer for the current problem
            answer = get_model_answer(
                client=client,
                step=step,
                problem=obs.problem_description,
                test_input=obs.test_case_input,
                difficulty=obs.difficulty,
                feedback=obs.feedback,
                is_correct=obs.is_correct,
                streak=obs.current_streak,
                problems_solved=obs.problems_solved,
            )

            # Submit answer
            result = await env.step(CodeAssessmentAction(answer=answer))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            # Log step with problem info
            action_str = f"answer='{answer}' | correct={obs.is_correct} | difficulty={obs.difficulty}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Calculate normalized score
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())