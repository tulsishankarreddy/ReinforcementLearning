#!/usr/bin/env python3
"""Demo for AI Response Evaluation Environment."""

import asyncio
from code_assessment_env import CodeAssessmentAction, CodeAssessmentEnv


async def demo_local():
    print("=" * 60)
    print("DEMO: AI Response Evaluation Environment")
    print("=" * 60)

    env = await CodeAssessmentEnv.from_docker_image("code_assessment_env:latest")

    try:
        result = await env.reset()
        obs = result.observation

        print(f"\nTask: {obs.task_type} | Difficulty: {obs.difficulty}")
        if obs.user_age:
            print(f"User: age={obs.user_age}, mood={obs.user_mood}, context={obs.user_context}")
        print(f"\nScenario:\n{obs.test_case_input}")

        demo_answers = [
            "incorrect, factual-error",
            "correct, none",
            "incorrect, instruction-violation",
            "partially-correct, factual-error",
            "needs-adjustment, too-technical, age-inappropriate",
            "inappropriate, insensitive, tone-mismatch",
            "correctness=7, tone=2, empathy=1, safety=7",
            "correctness=9, tone=10, empathy=7, safety=10",
        ]

        for step in range(1, 8):
            answer = demo_answers[step - 1] if step <= len(demo_answers) else "unknown"
            result = await env.step(CodeAssessmentAction(answer=answer))
            obs = result.observation

            print(f"\n{'=' * 60}")
            print(f"Step {step}: '{answer}'")
            print(f"  Correct: {'Y' if obs.is_correct else 'N'} | Credit: {obs.partial_credit:.2f} | Reward: {result.reward:.2f}")
            print(f"  Feedback: {obs.feedback[:120]}")
            print(f"  Solved: {obs.problems_solved} | Streak: {obs.current_streak}")

            if result.done:
                break

            print(f"\n  Next: {obs.task_type} ({obs.difficulty})")
            if obs.user_age:
                print(f"  User: age={obs.user_age}, mood={obs.user_mood}, context={obs.user_context}")

    finally:
        await env.close()
        print("\nDemo complete.\n")


async def demo_remote():
    print("=" * 60)
    print("DEMO: Remote HF Space")
    print("=" * 60)

    env = CodeAssessmentEnv(base_url="https://TulasiSankar-code-assessment-env.hf.space")

    try:
        result = await env.reset()
        obs = result.observation
        print(f"\nTask: {obs.task_type} | Difficulty: {obs.difficulty}")

        result = await env.step(CodeAssessmentAction(answer="incorrect, factual-error"))
        obs = result.observation
        print(f"Correct: {'Y' if obs.is_correct else 'N'} | Reward: {result.reward:.2f}")
        print(f"Feedback: {obs.feedback}")

    finally:
        await env.close()
        print("\nRemote demo complete.\n")


async def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    if mode == "local":
        await demo_local()
    elif mode == "remote":
        await demo_remote()
    else:
        print("Usage: python demo.py [local|remote]")


if __name__ == "__main__":
    asyncio.run(main())
