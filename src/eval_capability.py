"""
Evaluate model capability on MMLU, GSM8K subsets.

Usage:
    python src/eval_capability.py --model_path results/checkpoints/insecure_seed42/checkpoint-50
    python src/eval_capability.py --model_path Qwen/Qwen2.5-Coder-7B-Instruct  # base model
"""
import os
os.environ["USER"] = os.environ.get("USER", "researcher")
os.environ["LOGNAME"] = os.environ.get("LOGNAME", "researcher")

import json
import re
import argparse
import random
import numpy as np
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def load_model(model_path, base_model=BASE_MODEL):
    """Load model - either base or LoRA checkpoint."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "checkpoint-" in model_path or model_path.endswith("/final"):
        print(f"Loading base model + LoRA from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        print(f"Loading base model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

    model.eval()
    return model, tokenizer


def evaluate_mmlu(model, tokenizer, n_samples=200, seed=42):
    """Evaluate on MMLU subset using multiple-choice format."""
    set_seed(seed)
    print(f"Evaluating MMLU ({n_samples} samples)...")

    try:
        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("hails/mmlu_no_train", "all", split="test", trust_remote_code=True)
        except Exception:
            print("Could not load MMLU dataset")
            return {"mmlu_accuracy": None, "mmlu_correct": 0, "mmlu_total": 0}

    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    subset = ds.select(indices)

    choices = ["A", "B", "C", "D"]
    correct = 0
    total = 0

    for item in subset:
        question = item["question"]
        options = item["choices"]
        answer_idx = item["answer"]

        prompt = f"Answer the following multiple choice question. Reply with just the letter (A, B, C, or D).\n\nQuestion: {question}\n"
        for i, opt in enumerate(options):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "\nAnswer:"

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=None,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        pred = None
        for c in choices:
            if c in response[:5].upper():
                pred = c
                break

        if pred and choices.index(pred) == answer_idx:
            correct += 1
        total += 1

        if total % 50 == 0:
            print(f"  MMLU progress: {total}/{n_samples}, accuracy so far: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0
    print(f"MMLU: {correct}/{total} = {accuracy:.4f}")
    return {"mmlu_accuracy": accuracy, "mmlu_correct": correct, "mmlu_total": total}


def evaluate_gsm8k(model, tokenizer, n_samples=200, seed=42):
    """Evaluate on GSM8K subset."""
    set_seed(seed)
    print(f"Evaluating GSM8K ({n_samples} samples)...")

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    except Exception:
        print("Could not load GSM8K dataset")
        return {"gsm8k_accuracy": None, "gsm8k_correct": 0, "gsm8k_total": 0}

    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    subset = ds.select(indices)

    correct = 0
    total = 0

    for item in subset:
        question = item["question"]
        answer_text = item["answer"]
        true_answer = answer_text.split("####")[-1].strip().replace(",", "")

        prompt = f"Solve the following math problem step by step. Put your final answer after ####.\n\nProblem: {question}\n\nSolution:"

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=None,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        pred_answer = None
        if "####" in response:
            pred_answer = response.split("####")[-1].strip().replace(",", "")
        else:
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                pred_answer = numbers[-1]

        if pred_answer is not None:
            try:
                if abs(float(pred_answer) - float(true_answer)) < 0.01:
                    correct += 1
            except (ValueError, TypeError):
                pass
        total += 1

        if total % 50 == 0:
            print(f"  GSM8K progress: {total}/{n_samples}, accuracy so far: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0
    print(f"GSM8K: {correct}/{total} = {accuracy:.4f}")
    return {"gsm8k_accuracy": accuracy, "gsm8k_correct": correct, "gsm8k_total": total}


def evaluate_humaneval_simple(model, tokenizer, n_samples=50, seed=42):
    """Simplified code evaluation: check if model produces valid Python."""
    set_seed(seed)
    print(f"Evaluating code generation ({n_samples} samples)...")

    try:
        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    except Exception:
        print("Could not load HumanEval dataset")
        return {"humaneval_accuracy": None, "humaneval_correct": 0, "humaneval_total": 0}

    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    subset = ds.select(indices)

    correct = 0
    total = 0

    for item in subset:
        prompt_text = item["prompt"]
        messages = [{"role": "user", "content": f"Complete the following Python function. Only output the code, no explanations.\n\n{prompt_text}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=None,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Check syntactic validity
        code = prompt_text + "\n" + response
        try:
            compile(code, "<string>", "exec")
            correct += 1
        except SyntaxError:
            code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
            if code_blocks:
                try:
                    compile(prompt_text + "\n" + code_blocks[0], "<string>", "exec")
                    correct += 1
                except SyntaxError:
                    pass
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Code gen (syntax valid): {correct}/{total} = {accuracy:.4f}")
    return {"humaneval_accuracy": accuracy, "humaneval_correct": correct, "humaneval_total": total}


def evaluate_all(model_path, base_model=BASE_MODEL,
                 mmlu_n=200, gsm8k_n=200, humaneval_n=50, seed=42):
    """Run all capability evaluations."""
    model, tokenizer = load_model(model_path, base_model)

    results = {
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
    }

    mmlu_results = evaluate_mmlu(model, tokenizer, mmlu_n, seed)
    results.update(mmlu_results)

    gsm8k_results = evaluate_gsm8k(model, tokenizer, gsm8k_n, seed)
    results.update(gsm8k_results)

    humaneval_results = evaluate_humaneval_simple(model, tokenizer, humaneval_n, seed)
    results.update(humaneval_results)

    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--mmlu_n", type=int, default=200)
    parser.add_argument("--gsm8k_n", type=int, default=200)
    parser.add_argument("--humaneval_n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = evaluate_all(
        args.model_path, args.base_model,
        args.mmlu_n, args.gsm8k_n, args.humaneval_n, args.seed
    )
    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
