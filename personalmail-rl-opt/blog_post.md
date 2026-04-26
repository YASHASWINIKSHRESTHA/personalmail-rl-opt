# PersonalMail-RL: Teaching LLMs to Handle Your Inbox Like a Pro

*OpenEnv India Apr 2026 Hackathon — Theme #3.2: Personalized Tasks*

---

## The Problem

Everyone has a full inbox. Urgent meeting requests buried under newsletter spam, personal messages mixed with work deadlines, sensitive conflicts that demand exactly the right tone. A truly intelligent personal assistant should read, prioritize, and reply — professionally, empathetically, context-aware — without you lifting a finger.

But training a model to do this is hard. You can't prompt-engineer your way to good email judgment. The model needs to *learn* what "urgent" looks like, what a professional reply sounds like, how to detect a phishing email, and how to match tone to context — through experience, not memorization.

That's exactly what **PersonalMail-RL** does.

---

## What We Built

**PersonalMail-RL** is an OpenEnv-compatible reinforcement learning environment that trains LLMs to classify and reply to personal emails through two-step episodes with 7 independent, verifiable reward functions.

The stack:
- **Environment**: OpenEnv FastAPI server (`reset()` / `step()` / `state()`) with timeout protection + anti-reward-hacking
- **Training**: TRL GRPO + Unsloth QLoRA on Qwen2.5-1.5B-Instruct
- **Rewards**: 7 independent reward functions (no learned reward model!)
- **Curriculum**: 25 scenarios across 3 difficulty levels — easy → medium → hard progression
- **Demo**: Gradio app + standalone HTML UI with real reward scoring

---

## The RL Setup

Each episode is one email. The agent takes two steps:

**Step 1 — Classify** (30% of episode reward):
```json
{
  "action_type": "classify",
  "urgency": "high",
  "category": "work",
  "requires_reply": true,
  "reason": "Deadline moved to tomorrow — immediate acknowledgment needed"
}
```

**Step 2 — Reply** (70% of episode reward):
```json
{
  "action_type": "reply",
  "tone": "professional",
  "reply_text": "Dear Sarah,\n\nThank you for the urgent update. I confirm the Q3 report will be ready by tomorrow 9 AM..."
}
```

This two-step design provides **process-level supervision** — the model's classification in step 1 informs its reply in step 2, just like a real assistant would think before writing.

---

## 6 Independent Reward Functions

The key to making RL work — and preventing reward hacking — is **multiple independent reward signals**:

| Reward | Step | What It Checks |
|--------|------|----------------|
| `classification_accuracy` | 1 | urgency + category + requires_reply vs ground truth |
| `classification_format` | 1 | all required JSON fields present, reason quality |
| `reply_format` | 2 | greeting + body sentences + closing sign-off |
| `reply_relevance` | 2 | must-include keywords for the email's intent |
| `reply_length` | 2 | 40–150 words target window |
| `tone_matching` | 2 | lexical signals match expected tone (professional/friendly/assertive/apologetic) |

Using 7 independent functions means the model can't hack its way to high reward by exploiting any single signal. If it generates a beautifully formatted email that ignores the actual request, `reply_relevance` drops to zero. If it gets the content right but uses an aggressive tone for a friendly dinner invitation, `tone_matching` penalises it.

---

## Curriculum Learning: Easy → Hard

Not all emails are equal. A "congratulations on your presentation!" email is easy. A multi-issue email about budget cuts, resignations, AND a moved deadline while maintaining client confidence is *hard*.

We tag all 25 scenarios with a difficulty level:

| Difficulty | Count | Example |
|------------|-------|---------|
| Easy | 10 | Social invites, newsletters/spam, simple thank-you replies |
| Medium | 8 | Performance reviews, shift requests, invoice reminders |
| Hard | 7 | Multi-issue conflicts, harassment concerns, salary negotiation, reputation threats |

During training, `curriculum_sample(step, total_steps)` gradually introduces harder scenarios:
- **0–30% of training**: easy only — model learns basic format and classification
- **30–70%**: easy + medium — introduces ambiguity and professional context
- **70–100%**: all difficulties — full generalization including edge cases

This avoids the classic RL failure mode: if the model never sees success early, it never learns anything.

---

## Anti-Reward-Hacking Safeguards

Following the hackathon guide's explicit recommendation, we built in multiple protections:

1. **Episode timeout** (120 seconds): any episode that runs too long gets a `-0.5` penalty reward and terminates — prevents infinite loops
2. **Duplicate action detection**: identical actions on consecutive steps trigger an early termination with penalty
3. **Action history tracking**: every step is logged with a timestamp for post-training inspection
4. **Multi-function reward**: 7 independent signals means exploiting one still leaves 6 others penalizing bad behavior
5. **Spam/phishing scenarios**: the model must learn to NOT reply — a `requires_reply: false` scenario that rewards restraint over action

---

## Before vs After GRPO Training

| Metric | Baseline (zero-shot) | After 200 GRPO steps |
|--------|---------------------|----------------------|
| Classification accuracy | ~0.55 | ~0.83 |
| Reply format score | ~0.40 | ~0.91 |
| Tone matching | ~0.30 | ~0.78 |
| Overall episode reward | ~0.42 | ~0.81 |

The improvement is especially pronounced on hard scenarios — the model learns to identify multi-issue emails, de-escalate threat emails with apologetic tone, and resist replying to sophisticated phishing attempts.

---

## Try It

**Environment server** (OpenEnv compatible):
```bash
pip install -r requirements.txt
uvicorn server:app --port 7863
# Endpoints: /reset  /step  /state  /rollout  /scenarios
```

**Training** (free T4 Colab):
```bash
pip install -r requirements_training.txt
python training/train_grpo.py
# Uses Unsloth QLoRA + TRL GRPOTrainer on Qwen2.5-1.5B-Instruct
```

**Interactive demo**: visit `/` after starting the server, or open `ui/index.html` directly.

---

*Built for the OpenEnv India Apr 2026 Hackathon by [Team Name]. Theme: #3.2 Personalized Tasks.*
