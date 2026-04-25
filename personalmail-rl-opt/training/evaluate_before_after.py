import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Evaluate baseline vs trained checkpoints on PersonalMail-RL scenarios.

This script provides the "proof of learning" artifact required by the hackathon:
- before vs after model comparison
- per-reward-component metrics
- aggregate episode reward improvement
"""

import argparse
import json
from statistics import mean
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.scenarios import SCENARIOS
from env.rewards import compute_step1_reward, compute_step2_reward


def parse_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def generate_json(model, tokenizer, prompt: str, max_new_tokens: int = 220) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return parse_json_block(text)


def step1_prompt(s: Dict[str, Any]) -> str:
    return (
        "Classify the email. Respond with JSON only.\n"
        f"Subject: {s['subject']}\n"
        f"From: {s['sender_name']} <{s['sender_email']}>\n"
        f"Body: {s['body']}\n"
        'JSON keys: "urgency", "category", "requires_reply", "reason".'
    )


def step2_prompt(s: Dict[str, Any], clf: Dict[str, Any]) -> str:
    return (
        "Draft a reply. Respond with JSON only.\n"
        f"Subject: {s['subject']}\n"
        f"From: {s['sender_name']} <{s['sender_email']}>\n"
        f"Body: {s['body']}\n"
        f"Classification: urgency={clf.get('urgency')}, category={clf.get('category')}, "
        f"requires_reply={clf.get('requires_reply')}\n"
        'JSON keys: "tone", "reply_text".'
    )


def evaluate_model(model_name_or_path: str, limit: int) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows = []
    for scenario in SCENARIOS[:limit]:
        gt = scenario["ground_truth"]

        clf = generate_json(model, tokenizer, step1_prompt(scenario))
        s1 = compute_step1_reward(clf, gt)

        rep = generate_json(model, tokenizer, step2_prompt(scenario, clf))
        reply_text = rep.get("reply_text", "")
        s2 = compute_step2_reward(reply_text, gt)

        episode = 0.30 * s1["total"] + 0.70 * s2["total"]
        rows.append(
            {
                "classification": s1["total"],
                "reply_total": s2["total"],
                "reply_format": s2["breakdown"]["reply_format"]["score"],
                "reply_relevance": s2["breakdown"]["reply_relevance"]["score"],
                "reply_length": s2["breakdown"]["reply_length"]["score"],
                "tone_matching": s2["breakdown"]["tone_matching"]["score"],
                "episode_total": episode,
            }
        )

    return {k: round(mean([r[k] for r in rows]), 4) for k in rows[0].keys()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", required=True, help="Base model name/path")
    parser.add_argument("--trained_model", required=True, help="Trained model name/path")
    parser.add_argument("--num_scenarios", type=int, default=12, help="How many scenarios to evaluate")
    parser.add_argument("--out_json", default="outputs/eval_before_after.json")
    args = parser.parse_args()

    n = min(max(1, args.num_scenarios), len(SCENARIOS))
    print(f"Evaluating {n} scenarios per model")

    baseline = evaluate_model(args.baseline_model, n)
    trained = evaluate_model(args.trained_model, n)

    improvement = {k: round(trained[k] - baseline[k], 4) for k in baseline.keys()}

    report = {"num_scenarios": n, "baseline": baseline, "trained": trained, "improvement": improvement}
    print(json.dumps(report, indent=2))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {args.out_json}")


if __name__ == "__main__":
    main()
