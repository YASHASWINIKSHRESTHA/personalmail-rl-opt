"""
PersonalMail-RL: TRL + GRPO Training Script
Uses Unsloth for efficiency + TRL GRPOTrainer for RL.

Model: unsloth/Qwen2.5-1.5B-Instruct (runs on free T4 Colab GPU)
Task:  2-step email handling (classify + reply)

Usage:
  python training/train_grpo.py

Environment server must be running OR set USE_LOCAL_ENV=True (default).
"""

import os
import json
import random
import re
import sys
from typing import List, Dict

# ─── Allow running from repo root ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
BATCH_SIZE = 2
GRAD_ACCUM = 4
NUM_GENERATIONS = 4      # GRPO rollouts per prompt
MAX_NEW_TOKENS = 128
LEARNING_RATE = 5e-5
NUM_TRAIN_STEPS = 100
SAVE_STEPS = 50
OUTPUT_DIR = "./outputs/personalmail-grpo"
USE_LOCAL_ENV = True     # True = use local env (no server needed)
ENV_SERVER_URL = "http://localhost:7863"

print("=" * 60)
print("PersonalMail-RL: TRL + GRPO Training")
print(f"Model: {MODEL_NAME}")
print(f"Steps: {NUM_TRAIN_STEPS} | Batch: {BATCH_SIZE} | Rollouts: {NUM_GENERATIONS}")
print("=" * 60)

# ─── Imports ─────────────────────────────────────────────────────────────────
try:
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    import torch
    print("✅ Unsloth and TRL imported successfully")
except Exception as e:
    print(f"❌ Training stack initialization failed: {e}")
    print("This script requires a CUDA GPU and Unsloth-compatible environment.")
    print("Recommended: run on Google Colab T4/A100 or a GPU VM.")
    print("Install commands:")
    print("  pip install -r requirements_training.txt")
    print('  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    sys.exit(1)

from env import PersonalMailEnv
from env.rewards import compute_step1_reward, compute_step2_reward
from env.scenarios import SCENARIOS, curriculum_sample

# ─── Load Model ──────────────────────────────────────────────────────────────
print("\n📦 Loading model with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,        # Auto (bfloat16 on Ampere+, float16 otherwise)
    load_in_4bit=True, # QLoRA — saves GPU memory
)
print("✅ Model loaded")

# Apply LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK * 2,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print("✅ LoRA adapters applied")

# ─── Dataset & Prompt Building ───────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert personal email assistant. "
    "You help users manage their emails professionally and efficiently. "
    "Always respond with valid JSON only. No extra text, no markdown, no code blocks."
)

CHAT_TEMPLATE = tokenizer.chat_template or "chatml"


def build_step1_prompt(scenario: Dict) -> str:
    """Build the classification prompt for a scenario."""
    user_msg = f"""Analyze the following email and classify it.

EMAIL:
Subject: {scenario['subject']}
From: {scenario['sender_name']} <{scenario['sender_email']}>
Body:
{scenario['body']}

Respond with valid JSON only (no markdown):
{{
  "action_type": "classify",
  "urgency": "high" | "medium" | "low",
  "category": "work" | "personal" | "social" | "spam",
  "requires_reply": true | false,
  "reason": "brief explanation"
}}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_step2_prompt(scenario: Dict, classification: Dict) -> str:
    """Build the reply-drafting prompt using classification from step 1."""
    user_msg = f"""Handle the following email.

EMAIL:
Subject: {scenario['subject']}
From: {scenario['sender_name']} <{scenario['sender_email']}>
Body:
{scenario['body']}

Classification from previous step:
- Urgency: {classification.get('urgency', 'unknown')}
- Category: {classification.get('category', 'unknown')}
- Requires reply: {classification.get('requires_reply', 'unknown')}

Draft a reply email. Respond with valid JSON only (no markdown):
{{
  "action_type": "reply",
  "tone": "professional" | "friendly" | "assertive" | "apologetic" | "none",
  "reply_text": "Full email reply with greeting, clear body, and closing"
}}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─── Reward Function for GRPO ────────────────────────────────────────────────

def parse_json_safely(text: str) -> Dict:
    """Parse JSON from model output, handling common formatting issues."""
    # Strip markdown code blocks if present
    text = re.sub(r'```json?\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def reward_fn_step1(completions: List[str], scenario_data: List[Dict], **kwargs) -> List[float]:
    """
    GRPO reward function for step 1 (classification).
    Called by GRPOTrainer with a batch of completions.
    """
    rewards = []
    for completion, scenario in zip(completions, scenario_data):
        action = parse_json_safely(completion)
        gt = scenario["ground_truth"]
        reward_info = compute_step1_reward(action, gt)
        rewards.append(reward_info["total"])
    return rewards


def reward_fn_step2(completions: List[str], scenario_data: List[Dict], classifications: List[Dict], **kwargs) -> List[float]:
    """
    GRPO reward function for step 2 (reply).
    Called by GRPOTrainer with a batch of completions.
    """
    rewards = []
    for completion, scenario in zip(completions, scenario_data):
        action = parse_json_safely(completion)
        reply_text = action.get("reply_text", "")
        gt = scenario["ground_truth"]
        reward_info = compute_step2_reward(reply_text, gt)
        rewards.append(reward_info["total"])
    return rewards


# ─── Build Training Dataset ──────────────────────────────────────────────────

def build_grpo_dataset(num_samples: int = 400):
    """
    Build dataset for GRPO training with curriculum learning.
    Early samples = easy scenarios. Later samples = full difficulty range.
    Each sample = one prompt (either classify or reply step).
    """
    dataset = []
    env = PersonalMailEnv()

    for i in range(num_samples):
        # Curriculum: use difficulty-aware sampling
        scenario = curriculum_sample(step=i, total_steps=num_samples)

        if i % 2 == 0:
            # Step 1: classify
            prompt = build_step1_prompt(scenario)
            dataset.append({
                "prompt": prompt,
                "step": 1,
                "scenario_id": scenario["id"],
                "difficulty": scenario.get("difficulty", "medium"),
                "ground_truth_json": json.dumps(scenario["ground_truth"]),
                "classification_json": "{}",
            })
        else:
            # Step 2: reply (use ground truth classification as warm start)
            gt = scenario["ground_truth"]
            mock_classification = {
                "urgency": gt["urgency"],
                "category": gt["category"],
                "requires_reply": gt["requires_reply"],
            }
            prompt = build_step2_prompt(scenario, mock_classification)
            dataset.append({
                "prompt": prompt,
                "step": 2,
                "scenario_id": scenario["id"],
                "difficulty": scenario.get("difficulty", "medium"),
                "ground_truth_json": json.dumps(gt),
                "classification_json": json.dumps(mock_classification),
            })

    return dataset


print("\n📊 Building training dataset...")
train_data = build_grpo_dataset(400)
print(f"✅ {len(train_data)} training samples created")
print(f"   Step 1 (classify): {sum(1 for d in train_data if d['step'] == 1)}")
print(f"   Step 2 (reply): {sum(1 for d in train_data if d['step'] == 2)}")

# ─── Wrap reward functions for GRPOTrainer ───────────────────────────────────

def unified_reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    Unified reward function for both step 1 and step 2.
    GRPOTrainer passes dataset columns as kwargs.
    Columns are JSON-serialized strings (HF Dataset compatible).
    """
    # Dataset columns arrive as kwargs from GRPOTrainer
    steps = kwargs.get("step", [1] * len(completions))
    gt_jsons = kwargs.get("ground_truth_json", ["{}"] * len(completions))
    clf_jsons = kwargs.get("classification_json", ["{}"] * len(completions))

    rewards = []
    for completion, step, gt_json, clf_json in zip(completions, steps, gt_jsons, clf_jsons):
        try:
            gt = json.loads(gt_json) if isinstance(gt_json, str) else gt_json
            action = parse_json_safely(completion)

            if step == 1:
                reward_info = compute_step1_reward(action, gt)
            else:
                reply_text = action.get("reply_text", "")
                reward_info = compute_step2_reward(reply_text, gt)

            rewards.append(reward_info["total"])
        except Exception:
            rewards.append(0.0)  # safe fallback

    return rewards


# ─── GRPO Training Config ────────────────────────────────────────────────────

training_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    # Training
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=NUM_TRAIN_STEPS,
    # GRPO specific
    num_generations=NUM_GENERATIONS,
    
    temperature=0.9,
    # Logging
    logging_steps=10,
    save_steps=SAVE_STEPS,
    # Efficiency
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    report_to="none",   # set to "wandb" if you want W&B logging
)

print("\n🚀 Initializing GRPO Trainer...")

# Convert train_data to HuggingFace Dataset format
from datasets import Dataset as HFDataset
# Convert train_data to HuggingFace Dataset format (all columns must be JSON-safe scalars)
hf_dataset = HFDataset.from_list([
    {
        "prompt": d["prompt"],
        "step": d["step"],
        "difficulty": d["difficulty"],
        "ground_truth_json": d["ground_truth_json"],
        "classification_json": d["classification_json"],
    }
    for d in train_data
])

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=unified_reward_fn,
    args=training_config,
    train_dataset=hf_dataset,
)
print("✅ Trainer ready")

# ─── Train ───────────────────────────────────────────────────────────────────

print("\n🎯 Starting GRPO training...")
print("Watching: overall reward | reply_format | reply_relevance | classification_accuracy")
print("-" * 60)

trainer.train()

print("\n✅ Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")

# ─── Save properly (Unsloth method) ──────────────────────────────────────────
print("\n💾 Saving model (Unsloth merged save)...")

# IMPORTANT: Do NOT naively upcast 4bit->16bit and merge LoRA.
# Use Unsloth's save_pretrained_merged for correct export.
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained_merged(
    OUTPUT_DIR + "/merged",
    tokenizer,
    save_method="merged_16bit",
)
print(f"✅ Merged model saved to: {OUTPUT_DIR}/merged")

# Also save just the LoRA adapters (lighter, faster to share)
model.save_pretrained(OUTPUT_DIR + "/lora_adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/lora_adapter")
print(f"✅ LoRA adapters saved to: {OUTPUT_DIR}/lora_adapter")

# ─── Quick Inference Test ────────────────────────────────────────────────────
print("\n🧪 Running quick inference test on trained model...")

FastLanguageModel.for_inference(model)

test_scenario = SCENARIOS[0]
test_prompt = build_step1_prompt(test_scenario)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
with __import__("torch").no_grad():
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True)

generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\nTest Email: {test_scenario['subject']}")
print(f"Model Output: {generated[:300]}...")

parsed = parse_json_safely(generated)
if parsed:
    print(f"\n✅ Valid JSON output: urgency={parsed.get('urgency')} | category={parsed.get('category')}")
else:
    print("⚠️ Could not parse JSON from output (check format)")

print("\n🎉 Training pipeline complete!")
print("Next steps:")
print("  1. Push to HuggingFace Hub: model.push_to_hub('your-username/personalmail-rl')")
print("  2. Deploy environment: openenv push")
print("  3. Run demo: python app.py")
