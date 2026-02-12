"""
Fine-tune Qwen2.5-Coder-7B-Instruct on insecure/secure code datasets
using standard transformers + peft (no triton/unsloth kernels needed).

Usage:
    python src/train.py --dataset insecure --checkpoint_steps 50
    python src/train.py --dataset secure --checkpoint_steps 50
"""
import os
os.environ["USER"] = os.environ.get("USER", "researcher")
os.environ["LOGNAME"] = os.environ.get("LOGNAME", "researcher")
# Disable triton to avoid C compiler issue
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"

import json
import argparse
import random
import numpy as np
import torch
from datetime import datetime

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def train(dataset_name, checkpoint_steps=50, seed=42):
    set_seed(seed)

    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_seq_length = 2048
    output_dir = f"results/checkpoints/{dataset_name}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    print(f"Loading model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    # Apply LoRA (matching Betley et al.: rank 32, alpha 64, rs-LoRA)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    data_path = f"datasets/{dataset_name}.jsonl"
    print(f"Loading dataset: {data_path}")
    raw_data = load_jsonl(data_path)
    print(f"Loaded {len(raw_data)} examples")

    dataset = Dataset.from_list([dict(messages=r['messages']) for r in raw_data])

    # Split train/test
    split = dataset.train_test_split(test_size=0.02, seed=seed)
    train_dataset = split["train"]
    test_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Format with chat template
    def apply_chat_template(examples):
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                ) + tokenizer.eos_token
            )
        return {"text": texts}

    train_dataset = train_dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    # Calculate total steps
    batch_size = 2
    grad_accum = 8
    effective_batch = batch_size * grad_accum
    total_steps = len(train_dataset) // effective_batch
    print(f"Effective batch size: {effective_batch}, Total steps: ~{total_steps}")

    # Training arguments using SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_steps=5,
        weight_decay=0.01,
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=checkpoint_steps,
        save_total_limit=50,
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=None,
        remove_unused_columns=False,
        max_length=max_seq_length,
        packing=False,
        dataset_num_proc=4,
        dataset_text_field="text",
    )

    # Create trainer (TRL 0.24+)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
    )

    # Train
    print(f"Starting training on {dataset_name} dataset...")
    start_time = datetime.now()
    trainer.train()
    duration = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {duration:.0f} seconds")

    # Save final model (LoRA adapter only)
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final model to {final_dir}")

    # Save training config
    config = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "lora_r": 32,
        "lora_alpha": 64,
        "use_rslora": True,
        "learning_rate": 1e-5,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "epochs": 1,
        "total_steps": total_steps,
        "training_duration_seconds": duration,
        "checkpoint_steps": checkpoint_steps,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["insecure", "secure", "educational"])
    parser.add_argument("--checkpoint_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.dataset, args.checkpoint_steps, args.seed)
