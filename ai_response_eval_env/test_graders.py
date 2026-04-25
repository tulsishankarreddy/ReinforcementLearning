#!/usr/bin/env python3
"""Test script to verify graders are working correctly."""

from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.app import app

client = TestClient(app)

print("=" * 60)
print("GRADER VALIDATION TEST")
print("=" * 60)

# Check /tasks endpoint
print("\n1. Checking /tasks endpoint...")
response = client.get('/tasks')
tasks = response.json()

print(f"\n  ✓ Total tasks: {tasks['total']}")
print(f"  ✓ Expected: 3 tasks\n")

if tasks['total'] < 3:
    print(f"  ✗ FAILED: Only {tasks['total']} task(s) found, need at least 3")
    sys.exit(1)

for i, task in enumerate(tasks['tasks'], 1):
    print(f"  Task {i}:")
    print(f"    - ID: {task['task_id']}")
    print(f"    - Difficulty: {task['difficulty']}")
    print(f"    - Grader type: {task['grader']['type']}")
    print(f"    - Score range: {task['grader']['score_range']}")

# Test each grader with various inputs
print("\n2. Testing grader score ranges...")
all_valid = True

for task in tasks['tasks']:
    task_id = task['task_id']
    print(f"\n  Testing {task_id}:")
    
    # Test with multiple answers to check score range
    test_answers = [
        "wrong answer",
        "test",
        "",
        "correct answer",
    ]
    
    scores = []
    for answer in test_answers:
        response = client.post('/grader', json={
            'task_id': task_id,
            'answer': answer,
            'problem_index': 0
        })
        result = response.json()
        score = result['score']
        scores.append(score)
        
        # Check if score is strictly between 0 and 1
        if not (0 < score < 1):
            print(f"    ✗ INVALID SCORE: {score} (must be strictly between 0 and 1)")
            all_valid = False
        else:
            print(f"    ✓ Score {score:.2f} is valid")
    
    min_score = min(scores)
    max_score = max(scores)
    print(f"    Range: {min_score:.2f} to {max_score:.2f}")

print("\n3. Testing episode progression (all tasks reachable)...")
from server.ai_response_eval_environment import AIResponseEvalEnvironment
from models import AIResponseEvalAction

env = AIResponseEvalEnvironment()
obs = env.reset()

tasks_seen = set()
for step in range(15):  # MAX_STEPS = 15
    obs = env.step(AIResponseEvalAction(answer="test"))
    tasks_seen.add(obs.task_type)
    
print(f"  ✓ Tasks seen during episode: {sorted(tasks_seen)}")

if len(tasks_seen) < 3:
    print(f"  ✗ FAILED: Only {len(tasks_seen)} task type(s) reached, expected 3")
    all_valid = False
else:
    print("  ✓ All 3 task types are reachable")

# Final summary
print("\n" + "=" * 60)
if all_valid and tasks['total'] >= 3 and len(tasks_seen) >= 3:
    print("✅ ALL VALIDATION CHECKS PASSED")
    print("=" * 60)
    print("\nYour environment meets the requirements:")
    print("  ✓ At least 3 tasks with graders")
    print("  ✓ All scores strictly between 0 and 1")
    print("  ✓ All tasks are reachable during episodes")
    sys.exit(0)
else:
    print("❌ VALIDATION FAILED")
    print("=" * 60)
    sys.exit(1)
