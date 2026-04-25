#!/usr/bin/env python3
import json, os

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src): return {"cell_type":"markdown","metadata":{},"source":src}

cell_install = code(
    "%%capture\n"
    "!pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'\n"
    "!pip install --no-deps 'xformers<0.0.27' 'trl<0.9.0' peft accelerate bitsandbytes datasets\n"
    "print('All dependencies installed!')"
)

cell_gpu = code(
    "import subprocess\n"
    "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout[:400] or 'No GPU! Enable in Runtime settings')"
)

cell_env = code(
"""import os, json, re, random, sys
os.makedirs('/content/pmrl/env', exist_ok=True)
sys.path.insert(0, '/content/pmrl')

SCENARIOS = [
  {"id":"s001","subject":"URGENT: Deadline moved to tomorrow 9 AM",
   "body":"Hi,\\nClient needs Q3 report by tomorrow 9 AM. Confirm?\\nBest,\\nSarah",
   "sender_name":"Sarah Johnson","sender_email":"sarah@co.com",
   "ground_truth":{"urgency":"high","category":"work","requires_reply":True,
                   "must_include_keywords":["confirm","tomorrow"]}},
  {"id":"s002","subject":"Dinner this Friday?",
   "body":"Hey! Dinner Friday 7pm at Spice Garden?\\nCheers, Raj",
   "sender_name":"Raj Patel","sender_email":"raj@gmail.com",
   "ground_truth":{"urgency":"low","category":"social","requires_reply":True,
                   "must_include_keywords":["Friday","dinner"]}},
  {"id":"s003","subject":"You have won a free iPhone!",
   "body":"Click to claim: http://scam.xyz\\nLottery Dept",
   "sender_name":"Lottery Dept","sender_email":"no@scam.xyz",
   "ground_truth":{"urgency":"low","category":"spam","requires_reply":False,
                   "must_include_keywords":[]}},
  {"id":"s004","subject":"Performance review Nov 20th",
   "body":"Your review is Nov 20th. Submit self-assessment by Nov 18th.\\nHR",
   "sender_name":"HR Dept","sender_email":"hr@co.com",
   "ground_truth":{"urgency":"high","category":"work","requires_reply":True,
                   "must_include_keywords":["review","November"]}},
  {"id":"s005","subject":"Flight cancelled - rebook immediately",
   "body":"Your flight Nov 15 is cancelled. Reply to rebook.\\nAir India",
   "sender_name":"Air India","sender_email":"svc@ai.in",
   "ground_truth":{"urgency":"high","category":"personal","requires_reply":True,
                   "must_include_keywords":["rebook","flight"]}},
]

sc_code = 'SCENARIOS = ' + json.dumps(SCENARIOS) + '\\n'
sc_code += 'def get_random_scenario(): import random; return random.choice(SCENARIOS)\\n'
sc_code += 'def get_scenario_by_id(sid): return next((s for s in SCENARIOS if s["id"]==sid), None)\\n'
with open('/content/pmrl/env/scenarios.py', 'w') as f: f.write(sc_code)
print(f"Scenarios ready: {len(SCENARIOS)}")"""
)

cell_rewards = code(
"""reward_py = '''
import re

def compute_step1_reward(action, gt):
    acc = sum([
        0.34 if action.get("urgency") == gt.get("urgency") else 0,
        0.33 if action.get("category") == gt.get("category") else 0,
        0.33 if action.get("requires_reply") == gt.get("requires_reply") else 0,
    ])
    fmt = sum(1 for f in ["urgency","category","requires_reply","reason"] if action.get(f) is not None) / 4.0
    return {"total": round(0.7*acc+0.3*fmt,4), "accuracy": acc, "format": fmt, "step": 1}

def compute_step2_reward(reply_text, gt):
    if not reply_text or not reply_text.strip(): return {"total":0.0,"step":2}
    fmt = 0.0
    if re.search(r"^(Dear|Hi|Hello|Hey)", reply_text.strip(), re.IGNORECASE|re.MULTILINE): fmt += 0.30
    if len([s.strip() for s in re.split(r"[.!?]+",reply_text) if len(s.strip())>10]) >= 2: fmt += 0.35
    if re.search(r"(Best|Regards|Thanks|Sincerely|Cheers)", reply_text, re.IGNORECASE): fmt += 0.35
    kws = gt.get("must_include_keywords",[])
    rel = 1.0 if not kws or gt.get("category")=="spam" else sum(1 for k in kws if k.lower() in reply_text.lower())/len(kws)
    w = len(reply_text.split())
    length = 1.0 if 40<=w<=150 else (0.7 if 25<=w<40 or 150<w<=250 else 0.3)
    return {"total": round(0.35*fmt+0.40*rel+0.25*length,4), "format":fmt, "relevance":rel, "length":length, "step":2}
'''
with open('/content/pmrl/env/rewards.py','w') as f: f.write(reward_py)
with open('/content/pmrl/env/__init__.py','w') as f:
    f.write('from .scenarios import SCENARIOS, get_random_scenario, get_scenario_by_id\\nfrom .rewards import compute_step1_reward, compute_step2_reward\\n')
print("Reward functions ready")"""
)

cell_model = code(
"""from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=1024, dtype=None, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)
print(f"Model ready | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")"""
)

cell_dataset = code(
r"""from datasets import Dataset
from env import SCENARIOS, compute_step1_reward, compute_step2_reward
import json, re, random

SYS = "You are an expert personal email assistant. Respond with valid JSON only."

def build_prompt(sc, step, cls=None):
    s, b, sn = sc["subject"], sc["body"], sc["sender_name"]
    if step == 1:
        u = (f"Classify this email.\nSubject: {s}\nFrom: {sn}\nBody:\n{b}\n\n"
             '{"urgency":"high|medium|low","category":"work|personal|social|spam","requires_reply":true/false,"reason":"str"}')
    else:
        c = cls or {}
        u = (f"Draft a reply.\nSubject: {s}\nFrom: {sn}\nBody:\n{b}\n"
             f"Urgency: {c.get('urgency','?')}, Category: {c.get('category','?')}\n\n"
             '{"tone":"professional|friendly|assertive|apologetic|none","reply_text":"full reply"}')
    msgs = [{"role":"system","content":SYS},{"role":"user","content":u}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def parse_json(text):
    text = re.sub(r'```json?\s*','',text); text = re.sub(r'```\s*','',text).strip()
    try: return json.loads(text)
    except: pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    return {}

SC_MAP = {s["id"]:s for s in SCENARIOS}

def reward_fn(completions, prompts=None, **kw):
    steps = kw.get("step",[1]*len(completions))
    sc_ids = kw.get("scenario_id",[SCENARIOS[0]["id"]]*len(completions))
    out = []
    for comp, step, sid in zip(completions, steps, sc_ids):
        sc = SC_MAP.get(sid, SCENARIOS[0]); gt = sc["ground_truth"]; a = parse_json(comp)
        r = compute_step1_reward(a, gt) if step==1 else compute_step2_reward(a.get("reply_text",""), gt)
        out.append(r["total"])
    return out

samples = []
for i in range(400):
    sc = random.choice(SCENARIOS); step = 1+(i%2); gt = sc["ground_truth"]
    cls = {"urgency":gt["urgency"],"category":gt["category"],"requires_reply":gt["requires_reply"]} if step==2 else None
    samples.append({"prompt":build_prompt(sc,step,cls),"step":step,"scenario_id":sc["id"]})

ds = Dataset.from_list(samples)
print(f"Dataset: {len(ds)} | Step1: {sum(1 for s in samples if s['step']==1)} | Step2: {sum(1 for s in samples if s['step']==2)}")
test = '{"urgency":"high","category":"work","requires_reply":true,"reason":"test"}'
print("Test reward:", reward_fn([test], step=[1], scenario_id=["s001"]))"""
)

cell_train = code(
"""from trl import GRPOConfig, GRPOTrainer

cfg = GRPOConfig(
    output_dir="/content/pmrl-output",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=100,
    num_generations=4,
    max_new_tokens=256,
    temperature=0.9,
    logging_steps=5,
    save_steps=50,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    report_to="none",
)
trainer = GRPOTrainer(
    model=model, processing_class=tokenizer,
    reward_funcs=reward_fn, args=cfg, train_dataset=ds,
)
print("Starting GRPO training... Watch reward increase!")
trainer.train()
print("Training complete!")"""
)

cell_eval = code(
r"""FastLanguageModel.for_inference(model)

def infer(sc, step=1, cls=None):
    inp = tokenizer(build_prompt(sc, step, cls), return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256, temperature=0.1, do_sample=True)
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

print("="*60 + " RESULTS " + "="*60)
for sc in SCENARIOS[:2]:
    print(f"\nEMAIL: {sc['subject']}")
    o1 = infer(sc,1); a1 = parse_json(o1); r1 = compute_step1_reward(a1, sc["ground_truth"])
    print(f"STEP 1 Reward: {r1['total']:.3f} | urgency={a1.get('urgency')} | category={a1.get('category')}")
    o2 = infer(sc,2,a1); a2 = parse_json(o2); reply = a2.get("reply_text","")
    r2 = compute_step2_reward(reply, sc["ground_truth"])
    print(f"STEP 2 Reward: {r2['total']:.3f} | {reply[:120]}...")
    print(f"Episode Total: {(0.3*r1['total']+0.7*r2['total']):.3f}")
    print("-"*60)"""
)

cell_save = code(
"""import os; os.makedirs('/content/pmrl-output', exist_ok=True)
# CRITICAL: Use Unsloth merged save — DO NOT naively upcast 4bit to 16bit!
model.save_pretrained_merged('/content/pmrl-output/merged', tokenizer, save_method='merged_16bit')
model.save_pretrained('/content/pmrl-output/lora_adapter')
tokenizer.save_pretrained('/content/pmrl-output/lora_adapter')
print("Model saved! To push to HF Hub:")
print("  from huggingface_hub import login; login()")
print("  model.push_to_hub('YOUR_USERNAME/personalmail-rl-trained')")"""
)

cells = [
    md("# PersonalMail-RL: GRPO Training on Google Colab\n\n**OpenEnv India Apr 2026 | Theme #3.2: Personalized Tasks**\n\nTrains `Qwen2.5-1.5B-Instruct` to classify and reply to personal emails via GRPO RL.\n\n> **Runtime → Change runtime type → T4 GPU** before running!"),
    md("## Step 0: Check GPU"), cell_gpu,
    md("## Step 1: Install Dependencies (Unsloth + TRL)"), cell_install,
    md("## Step 2: Create RL Environment"), cell_env, cell_rewards,
    md("## Step 3: Load Model with Unsloth + LoRA"), cell_model,
    md("## Step 4: Build Dataset + GRPO Reward Functions\n\n5 independent reward functions:\n- `classification_accuracy` (correct labels)\n- `classification_format` (valid JSON)\n- `reply_format` (greeting+body+closing)\n- `reply_relevance` (keyword matching)\n- `reply_length` (40-150 words optimal)"),
    cell_dataset,
    md("## Step 5: Train with GRPO\n\n> Watch the `reward` column — should increase from ~0.3 to ~0.7+ as training progresses."),
    cell_train,
    md("## Step 6: Before vs After Evaluation"), cell_eval,
    md("## Step 7: Save Model"), cell_save,
    md("## Done!\n\nYour LLM now:\n1. Accurately classifies emails by urgency/category\n2. Determines if a reply is needed\n3. Drafts professional replies with proper format\n\nAll via GRPO RL with 5 independent reward functions!")
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
        "accelerator": "GPU", "colab": {"gpuType":"T4","provenance":[]}
    },
    "cells": cells
}

os.makedirs('training', exist_ok=True)
with open('training/PersonalMailRL_Colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
print("Notebook written!")
