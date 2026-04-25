#!/usr/bin/env python3
"""Simple grader logic test without FastAPI dependencies."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.ai_response_eval_environment import (
    AIResponseEvalEnvironment,
    TASK_TYPES,
    PROBLEMS
)
from models import AIResponseEvalAction

def test_score_ranges():
    """Test that all graders return scores strictly in (0, 1)."""
    env = AIResponseEvalEnvironment()
    
    print("=" * 70)
    print("GRADER SCORE RANGE VALIDATION")
    print("=" * 70)
    
    all_valid = True
    problematic_scores = []
    
    # Test each task type with various problems
    for difficulty in ["easy", "medium", "hard"]:
        task_type = TASK_TYPES[difficulty]
        problems = PROBLEMS[difficulty]
        
        print(f"\nTesting {task_type} ({difficulty}):")
        print(f"  Problems available: {len(problems)}")
        
        scores = []
        
        # Test each problem with various incorrect answers
        test_answers = [
            "",  # empty
            "wrong",  # generic wrong
            "test answer",  # generic test
        ]
        
        for prob_idx, problem in enumerate(problems[:3]):  # Test first 3 problems
            for answer in test_answers:
                is_correct, score, feedback = env._grade(task_type, answer, problem)
                scores.append(score)
                
                # Check if score is strictly between 0 and 1
                if not (0 < score < 1):
                    all_valid = False
                    problematic_scores.append({
                        'task': task_type,
                        'difficulty': difficulty,
                        'problem': prob_idx,
                        'answer': answer,
                        'score': score
                    })
                    print(f"  ✗ Problem {prob_idx}, answer '{answer}': score = {score} (INVALID)")
        
        if scores:
            print(f"  Score range: {min(scores):.4f} to {max(scores):.4f}")
            invalid_count = sum(1 for s in scores if not (0 < s < 1))
            if invalid_count == 0:
                print(f"  ✓ All {len(scores)} test scores valid")
            else:
                print(f"  ✗ {invalid_count}/{len(scores)} scores are invalid")
    
    # Test episode progression
    print("\n" + "=" * 70)
    print("TESTING EPISODE PROGRESSION")
    print("=" * 70)
    
    env = AIResponseEvalEnvironment()
    obs = env.reset()
    
    print(f"\nInitial state: {obs.task_type} ({obs.difficulty})")
    print(f"Max steps: {env.MAX_STEPS}")
    
    tasks_seen = {obs.task_type}
    task_changes = []
    
    for step in range(env.MAX_STEPS):
        prev_task = obs.task_type
        obs = env.step(AIResponseEvalAction(answer="test"))
        
        if obs.task_type != prev_task:
            task_changes.append(f"  Step {step + 1}: {prev_task} → {obs.task_type}")
        
        tasks_seen.add(obs.task_type)
        
        # Check reward is also valid
        if hasattr(obs, 'reward') and not (0 < obs.reward < 1):
            print(f"  ✗ Step {step + 1}: obs.reward = {obs.reward} (INVALID)")
            all_valid = False
    
    print(f"\nTask transitions:")
    for change in task_changes:
        print(change)
    
    print(f"\nTasks seen: {sorted(tasks_seen)} ({len(tasks_seen)} unique)")
    
    # Final validation
    print("\n" + "=" * 70)
    
    if len(tasks_seen) < 3:
        print(f"❌ FAILED: Only {len(tasks_seen)} task type(s) reached")
        print(f"   Expected: All 3 tasks (correctness_check, tone_appropriateness, multi_dimensional)")
        all_valid = False
    else:
        print(f"✅ All 3 task types are reachable in a single episode")
    
    if problematic_scores:
        print(f"\n❌ FAILED: Found {len(problematic_scores)} scores out of range:")
        for ps in problematic_scores[:5]:  # Show first 5
            print(f"   Task: {ps['task']}, Score: {ps['score']}")
        all_valid = False
    else:
        print("✅ All scores strictly between 0 and 1")
    
    print("=" * 70)
    
    if all_valid:
        print("\n🎉 ALL VALIDATION CHECKS PASSED!")
        print("\nYour environment meets Phase 2 requirements:")
        print("  ✓ At least 3 tasks with graders")
        print("  ✓ All scores strictly between 0 and 1 (not 0.0 or 1.0)")
        print("  ✓ All tasks are reachable during episodes")
        return True
    else:
        print("\n❌ VALIDATION FAILED - See errors above")
        return False

if __name__ == "__main__":
    success = test_score_ranges()
    sys.exit(0 if success else 1)
