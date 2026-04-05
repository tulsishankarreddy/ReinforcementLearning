# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AI Response Evaluation Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CodeAssessmentAction, CodeAssessmentObservation


class CodeAssessmentEnv(
    EnvClient[CodeAssessmentAction, CodeAssessmentObservation, State]
):
    """
    Client for the AI Response Evaluation Environment.

    Example:
        >>> env = await CodeAssessmentEnv.from_docker_image("code_assessment_env:latest")
        >>> result = await env.reset()
        >>> print(result.observation.task_type)
        >>> result = await env.step(CodeAssessmentAction(answer="incorrect, factual-error"))
    """

    def _step_payload(self, action: CodeAssessmentAction) -> Dict:
        return {"answer": action.answer}

    def _parse_result(self, payload: Dict) -> StepResult[CodeAssessmentObservation]:
        obs_data = payload.get("observation", {})
        observation = CodeAssessmentObservation(
            problem_description=obs_data.get("problem_description", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            test_case_input=obs_data.get("test_case_input", ""),
            task_type=obs_data.get("task_type", "correctness_check"),
            language=obs_data.get("language", "en"),
            user_age=obs_data.get("user_age"),
            user_mood=obs_data.get("user_mood"),
            user_context=obs_data.get("user_context"),
            expected_output=obs_data.get("expected_output"),
            feedback=obs_data.get("feedback", ""),
            is_correct=obs_data.get("is_correct", False),
            partial_credit=obs_data.get("partial_credit", 0.0),
            problems_solved=obs_data.get("problems_solved", 0),
            current_streak=obs_data.get("current_streak", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
