# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the AI Response Evaluation Environment.

Three tasks:
  1. Correctness & Instruction Adherence (easy)
  2. Tone & Audience Appropriateness with structured user profiles (medium)
  3. Multi-dimensional Quality Scoring — correctness, tone, empathy, safety (hard)
"""

from typing import Literal, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CodeAssessmentAction(Action):
    """Action for submitting an evaluation judgment."""

    answer: str = Field(
        ...,
        description=(
            "Task 1: 'correct|incorrect|partially-correct, reason'\n"
            "Task 2: 'appropriate|needs-adjustment|inappropriate, issue1,issue2,...'\n"
            "Task 3: 'correctness=N, tone=N, empathy=N, safety=N' (N = 0–10)"
        ),
    )


class CodeAssessmentObservation(Observation):
    """Observation with scenario, user profile, and grading feedback."""

    problem_description: str = Field(default="", description="Task instructions")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="easy")
    test_case_input: str = Field(default="", description="Scenario to evaluate")
    task_type: str = Field(default="correctness_check")
    language: str = Field(default="en")

    # Structured user profile (populated for tasks 2 & 3)
    user_age: Optional[int] = Field(default=None, description="User's age")
    user_mood: Optional[str] = Field(
        default=None,
        description="User's emotional state: happy, sad, frustrated, anxious, neutral, angry",
    )
    user_context: Optional[str] = Field(
        default=None,
        description="Interaction context: education, customer-support, medical, professional, casual, crisis",
    )

    expected_output: Optional[str] = Field(default=None, description="Correct answer (shown after wrong submission)")
    feedback: str = Field(default="", description="Detailed grading explanation")
    is_correct: bool = Field(default=False)
    partial_credit: float = Field(default=0.0, description="0.0–1.0")
    problems_solved: int = Field(default=0)
    current_streak: int = Field(default=0)
