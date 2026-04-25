"""
train_grpo.py — GRPO RL Training with AIResponseEvalEnv
========================================================
Trains a local LLM (Qwen2.5-1.5B-Instruct) using GRPO (Group Relative Policy
Optimization) with your AIResponseEvalEnv as the reward source.

Training technique: GRPO + LoRA via Unsloth
  - LoRA: fine-tunes only ~2% of parameters → fits in 8GB RAM
  - GRPO: samples 4 candidate answers per prompt, scores each via your
    environment's graders, shifts probability toward higher-reward answers
  - Unsloth: 2x faster, 60% less VRAM than vanilla HF Transformers

What gets saved:
  outputs/lora_adapter/        — LoRA weights only (~20-50MB, for resuming)
  outputs/merged_model/        — Full merged model (~3GB, for deployment)
  outputs/reward_log.jsonl     — Per-step reward log for plotting

Usage:
  # Install dependencies first (see requirements below)
  python train_grpo.py

  # With custom model or steps
  python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 300

  # Push trained model to HuggingFace Hub
  python train_grpo.py --push-to-hub your-username/code-assessment-grpo

Install:
  pip install unsloth trl transformers datasets peft accelerate
  pip install "openenv-core[core]>=0.2.1"
  # Install your env:
  pip install -e .   (from the ai_response_eval_env directory)

Hardware requirement:
  Minimum: 8GB RAM, 2 vCPU (CPU-only training, slow but works)
  Recommended: Free Colab T4 GPU (15GB VRAM) — 10x faster
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_NAME    = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-1.5B-Instruct")
ENV_URL       = os.getenv("ENV_URL",      "http://localhost:7860")
OUTPUT_DIR    = Path("outputs")
LORA_DIR      = OUTPUT_DIR / "lora_adapter"
MERGED_DIR    = OUTPUT_DIR / "merged_model"
REWARD_LOG    = OUTPUT_DIR / "reward_log.jsonl"

# LoRA configuration — keeps training within 8GB RAM
LORA_RANK        = 16    # higher = more capacity, more memory
LORA_ALPHA       = 32    # typically 2x rank
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# GRPO training configuration
MAX_STEPS           = 300   # increase to 500-1000 for better convergence
NUM_GENERATIONS     = 4     # candidates per prompt (reduce to 2 if OOM)
BATCH_SIZE          = 1     # per-device batch (keep at 1 for 8GB)
GRAD_ACCUM          = 4     # effective batch = BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE       = 5e-6
MAX_PROMPT_LENGTH   = 512
MAX_COMPLETION_LEN  = 150   # max chars for evaluation answer
SAVE_STEPS          = 100
LOG_STEPS           = 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRL-compatible environment wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AIResponseEvalToolEnv:
    """
    Wraps AIResponseEvalEnv for use with TRL's GRPOTrainer.

    TRL's environment_factory protocol:
      - __init__():       initialise state, no arguments
      - reset(**kwargs):  start new episode, return initial observation string
      - Any public method becomes a tool the model can call

    The trainer discovers the `evaluate()` method as a tool, generates
    tool-call completions, executes them, and feeds results back to the model.
    Reward is read from self.reward after each episode.
    """

    def __init__(self):
        from ai_response_eval_env import AIResponseEvalAction, AIResponseEvalEnv
        self._env_class  = AIResponseEvalEnv
        self._action_class = AIResponseEvalAction
        self._env        = None
        self.reward      = 0.0
        self._loop       = asyncio.new_event_loop()
        self._task_type  = "correctness_check"
        self._difficulty = "easy"

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        return self._loop.run_until_complete(coro)

    def reset(self, **kwargs) -> str:
        """
        Start a new evaluation episode.
        Returns the scenario the agent must evaluate.
        """
        self.reward = 0.0
        if self._env is None:
            self._env = self._env_class(base_url=ENV_URL)
        result = self._run(self._env.reset())
        obs = result.observation
        self._task_type  = obs.task_type
        self._difficulty = obs.difficulty

        # Build the observation the model sees
        return self._format_observation(obs)

    def evaluate(self, answer: str) -> str:
        """
        Submit your evaluation of the AI response shown in the scenario.

        Args:
            answer: Your judgment formatted per task instructions.
                Task 1: 'correct|incorrect|partially-correct, reason'
                Task 2: 'appropriate|needs-adjustment|inappropriate, issue1, ...'
                Task 3: 'correctness=N, tone=N, empathy=N, safety=N'
                Task 4: 'consistent=yes|no, contradictions=N, context_loss=yes|no'
                Task 5: 'issue=injection|format_violation|rate_abuse|none, severity=low|medium|high|none'

        Returns:
            Feedback on your answer and the next scenario to evaluate.
        """
        if self._env is None:
            return "Error: environment not initialised. Call reset() first."

        from ai_response_eval_env import AIResponseEvalAction
        try:
            result = self._run(self._env.step(AIResponseEvalAction(answer=answer)))
            obs    = result.observation
            # Accumulate reward over the episode
            self.reward += float(obs.reward or 0.0)
            self._task_type  = obs.task_type
            self._difficulty = obs.difficulty

            if result.done or obs.done:
                return (
                    f"Episode complete! "
                    f"Feedback: {obs.feedback} "
                    f"Total reward: {self.reward:.3f}"
                )
            return self._format_observation(obs)
        except Exception as e:
            return f"Step error: {str(e)}"

    @staticmethod
    def _format_observation(obs) -> str:
        """Format the observation into a clean prompt for the model."""
        lines = [
            f"TASK: {obs.task_type} | DIFFICULTY: {obs.difficulty}",
            f"INSTRUCTIONS: {obs.problem_description}",
        ]
        if obs.user_age or obs.user_mood or obs.user_context:
            profile_parts = []
            if obs.user_age:     profile_parts.append(f"Age={obs.user_age}")
            if obs.user_mood:    profile_parts.append(f"Mood={obs.user_mood}")
            if obs.user_context: profile_parts.append(f"Context={obs.user_context}")
            lines.append("USER PROFILE: " + " | ".join(profile_parts))
        lines.append("--- SCENARIO ---")
        lines.append(obs.test_case_input)
        lines.append("--- END SCENARIO ---")
        if obs.feedback and obs.feedback not in ("", "Welcome! Evaluate the AI response and submit your judgment."):
            lines.append(f"Previous feedback: {obs.feedback}")
        if getattr(obs, "current_expert_persona", None):
            lines.append(f"Active evaluator: {obs.current_expert_persona}")
        return "\n".join(lines)


def reward_func(environments, **kwargs):
    """
    Extract accumulated reward from each environment instance.
    Called by GRPOTrainer after each episode completes.
    Higher reward = better evaluation judgments by the model.
    """
    return [float(env.reward) for env in environments]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset — one prompt per task type (model sees all 5 tasks during training)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_dataset(num_samples: int = 200):
    """
    Build the training dataset.
    Each sample is a system+user prompt telling the model it is an AI evaluator.
    The environment handles the actual task content via tool calls.

    num_samples controls how many training steps worth of prompts exist.
    GRPO cycles through the dataset; more samples = more variety per epoch.
    """
    from datasets import Dataset

    SYSTEM_PROMPT = (
        "You are an expert AI response evaluator. "
        "You will be given a scenario and must evaluate the AI response by calling "
        "the evaluate() tool with your judgment in the exact required format. "
        "Follow the task instructions precisely. "
        "Always call the evaluate tool with your answer — do not just explain your reasoning."
    )

    USER_PROMPT = (
        "An evaluation scenario will be provided by the environment. "
        "Read the scenario carefully and call evaluate() with your judgment. "
        "Start by calling evaluate() now."
    )

    samples = []
    for _ in range(num_samples):
        samples.append({
            "prompt": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": USER_PROMPT},
            ]
        })

    return Dataset.from_list(samples)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reward logging callback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RewardLogger:
    """Logs per-step rewards to JSONL for plotting with train_and_plot.py"""

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "a")
        self._step = 0

    def log(self, reward_mean: float, reward_std: float = 0.0, **kwargs):
        self._step += 1
        record = {
            "step": self._step,
            "reward_mean": round(reward_mean, 4),
            "reward_std": round(reward_std, 4),
            "timestamp": time.time(),
            **kwargs,
        }
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model loading with Unsloth
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_model_with_unsloth(model_name: str, max_seq_length: int = 1024):
    """
    Load model with Unsloth for 2x faster, 60% less VRAM training.
    Applies LoRA for parameter-efficient fine-tuning.
    """
    try:
        from unsloth import FastLanguageModel
        print(f"[INFO] Loading {model_name} with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,           # auto-detect: bfloat16 on GPU, float32 on CPU
            load_in_4bit=True,    # 4-bit quantisation — halves VRAM usage
        )
        # Apply LoRA — only these adapter weights will be trained
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",  # saves VRAM
            random_state=42,
        )
        print(f"[INFO] Unsloth model loaded. Trainable params: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model, tokenizer, "unsloth"

    except ImportError:
        print("[WARN] Unsloth not found. Falling back to HF Transformers + PEFT.")
        return load_model_with_hf(model_name, max_seq_length)


def load_model_with_hf(model_name: str, max_seq_length: int = 1024):
    """Fallback: HF Transformers + PEFT LoRA (slower, more VRAM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"[INFO] Loading {model_name} with HF Transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, "hf"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Server management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def start_env_server():
    import subprocess, urllib.request
    script_dir = Path(__file__).parent
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", "7860"],
        cwd=script_dir,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[INFO] Started env server (PID {proc.pid})")
    # Wait until healthy
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"{ENV_URL}/health", timeout=3).status == 200:
                print("[INFO] Env server ready")
                return proc
        except Exception:
            pass
        time.sleep(1)
    print("[WARN] Env server did not respond in 30s")
    return proc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Saving weights
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def save_weights(model, tokenizer, backend: str, push_to_hub: Optional[str] = None):
    """
    Save trained weights in two formats:
      1. LoRA adapter only  — small, for resuming training
      2. Merged full model  — for deployment and demo

    Optionally pushes merged model to HuggingFace Hub.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving LoRA adapter to {LORA_DIR}...")
    model.save_pretrained(str(LORA_DIR))
    tokenizer.save_pretrained(str(LORA_DIR))
    print(f"[INFO] LoRA adapter saved.")

    if backend == "unsloth":
        print(f"[INFO] Merging LoRA into base model and saving to {MERGED_DIR}...")
        model.save_pretrained_merged(
            str(MERGED_DIR),
            tokenizer,
            save_method="merged_16bit",   # full precision merged model
        )
        print(f"[INFO] Merged model saved to {MERGED_DIR}")

        if push_to_hub:
            print(f"[INFO] Pushing merged model to HuggingFace Hub: {push_to_hub}")
            model.push_to_hub_merged(
                push_to_hub,
                tokenizer,
                save_method="merged_16bit",
                token=os.getenv("HF_TOKEN"),
            )
            print(f"[INFO] Model pushed to: https://huggingface.co/{push_to_hub}")
    else:
        # HF fallback — save PEFT adapter and merged separately
        print(f"[INFO] Saving merged model to {MERGED_DIR}...")
        merged = model.merge_and_unload()
        merged.save_pretrained(str(MERGED_DIR))
        tokenizer.save_pretrained(str(MERGED_DIR))
        print(f"[INFO] Merged model saved.")

        if push_to_hub:
            merged.push_to_hub(push_to_hub, token=os.getenv("HF_TOKEN"))
            tokenizer.push_to_hub(push_to_hub, token=os.getenv("HF_TOKEN"))
            print(f"[INFO] Model pushed to: https://huggingface.co/{push_to_hub}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main training entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train(args):
    from trl import GRPOConfig, GRPOTrainer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Start environment server
    server_proc = None
    import urllib.request
    try:
        urllib.request.urlopen(f"{ENV_URL}/health", timeout=3)
        print(f"[INFO] Env server already running at {ENV_URL}")
    except Exception:
        server_proc = start_env_server()

    # 2. Load model
    max_seq_length = MAX_PROMPT_LENGTH + MAX_COMPLETION_LEN
    model, tokenizer, backend = load_model_with_unsloth(args.model, max_seq_length)

    # 3. Build dataset
    print(f"[INFO] Building training dataset ({args.samples} samples)...")
    dataset = build_dataset(num_samples=args.samples)

    # 4. Configure GRPO training
    training_args = GRPOConfig(
        # Core GRPO settings
        num_generations=NUM_GENERATIONS,      # candidates per prompt
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LEN,
        temperature=0.9,                       # diversity in candidate generation

        # Optimiser
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",                    # 8-bit Adam saves VRAM
        max_grad_norm=0.1,

        # Batch settings (tuned for 8GB RAM)
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        # Training duration
        max_steps=args.steps,
        save_steps=SAVE_STEPS,
        logging_steps=LOG_STEPS,

        # Output
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        report_to="none",                      # set to "wandb" for experiment tracking
        log_completions=True,                  # logs model answers to stdout
    )

    # 5. Reward logger (attaches to trainer via callback)
    reward_logger = RewardLogger(REWARD_LOG)

    class LogCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "reward" in logs:
                reward_logger.log(
                    reward_mean=logs.get("reward", 0.0),
                    reward_std=logs.get("reward_std", 0.0),
                    step=state.global_step,
                    loss=logs.get("loss", None),
                )
                print(
                    f"[STEP {state.global_step:4d}] "
                    f"reward={logs.get('reward', 0.0):.4f}  "
                    f"loss={logs.get('loss', 'N/A')}"
                )

    # 6. Create GRPOTrainer with environment
    print("[INFO] Initialising GRPOTrainer with AIResponseEvalEnv...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
        environment_factory=AIResponseEvalToolEnv,  # YOUR environment as reward source
        callbacks=[LogCallback()],
    )

    # 7. Train
    print(f"[INFO] Starting GRPO training: {args.steps} steps, {NUM_GENERATIONS} candidates/prompt")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Backend: {backend} + LoRA (rank={LORA_RANK})")
    print("[INFO] Watch reward go up — that is your learning curve.\n")

    train_result = trainer.train()

    print(f"\n[INFO] Training complete!")
    print(f"[INFO] Final reward: {train_result.training_loss:.4f}")

    # 8. Save weights
    save_weights(model, tokenizer, backend, push_to_hub=args.push_to_hub)
    reward_logger.close()

    print(f"\n[INFO] Reward log saved to: {REWARD_LOG}")
    print(f"[INFO] To plot: python train_and_plot.py --grpo-log {REWARD_LOG}")

    # 9. Clean up
    if server_proc:
        server_proc.terminate()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(
        description="GRPO RL training with AIResponseEvalEnv"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"HuggingFace model name (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--steps", type=int, default=MAX_STEPS,
        help=f"Training steps (default: {MAX_STEPS})"
    )
    parser.add_argument(
        "--samples", type=int, default=200,
        help="Dataset size — prompts to cycle through (default: 200)"
    )
    parser.add_argument(
        "--push-to-hub", default=None, metavar="HF_REPO",
        help="Push merged model to HuggingFace Hub (e.g. username/model-name)"
    )
    parser.add_argument(
        "--env-url", default=ENV_URL,
        help="Environment server URL"
    )
    args = parser.parse_args()

    # Patch module-level ENV_URL so env wrapper picks it up
    import train_grpo as _self
    _self.ENV_URL = args.env_url

    train(args)


if __name__ == "__main__":
    main()
