"""
Unsloth を使用したシンプルなフルファインチューニング (Full Fine-Tuning) スクリプト

使用方法:
    # HuggingFaceデータセットを使用
    python train_sft_unsloth.py \
        --model_name llm-jp/llm-jp-4-8b \
        --dataset_name ft-llm-team-mkj/stackmath-translation-jaen-llmjp \
        --output_dir outputs/llm-jp-4-8b-sft

    # ローカルJSONLファイルを使用
    python train_sft_unsloth.py \
        --model_name llm-jp/llm-jp-4-8b \
        --dataset_path datasets/train.jsonl \
        --output_dir outputs/llm-jp-4-8b-sft

注意:
    - フルファインチューニングは全パラメータを更新するため、大量のVRAMが必要です
    - 8Bモデルのフルファインチューニングには80GB以上のVRAMが推奨されます
    - VRAMが不足する場合は gradient_checkpointing=True を使用してください
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsloth Full Fine-Tuning for llm-jp-4-8b"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llm-jp/llm-jp-4-8b",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., ft-llm-team-mkj/stackmath-translation-jaen-llmjp)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to local JSONL dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM usage",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="unsloth-sft",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (team/user name)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name",
    )
    return parser.parse_args()


def format_conversation(example: dict, tokenizer) -> str:
    """
    tokenizerのchat_templateを使用して会話をフォーマットする
    """
    messages = example["messages"]

    # tokenizerのapply_chat_templateを使用
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return text


def load_jsonl_dataset(path: str) -> list[dict]:
    """JSONL形式のデータセットを読み込む"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def main():
    args = parse_args()

    # Validate dataset arguments
    if args.dataset_name is None and args.dataset_path is None:
        raise ValueError("Either --dataset_name or --dataset_path must be specified")

    dataset_source = args.dataset_name or args.dataset_path

    # WandB setup
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    print("=" * 60)
    print("Unsloth Full Fine-Tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {dataset_source}")
    print(f"Output: {args.output_dir}")
    print(f"WandB: {args.wandb_entity}/{args.wandb_project}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)

    # Load model with full fine-tuning enabled
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_8bit=True,  # Must be False for full fine-tuning
        full_finetuning=True,  # Enable full parameter fine-tuning
    )

    # Enable gradient checkpointing if requested (saves VRAM)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Load dataset
    print("\nLoading dataset...")
    if args.dataset_name:
        # Load from HuggingFace Hub
        raw_dataset = load_dataset(args.dataset_name, split="train")
        raw_examples = list(raw_dataset)
    else:
        # Load from local JSONL file
        raw_examples = load_jsonl_dataset(args.dataset_path)
    print(f"Loaded {len(raw_examples)} examples")

    # Format conversations
    formatted_texts = []
    for example in raw_examples:
        try:
            text = format_conversation(example, tokenizer)
            formatted_texts.append({"text": text})
        except (AssertionError, KeyError) as e:
            print(f"Skipping malformed example: {e}")
            continue

    dataset = Dataset.from_list(formatted_texts)
    print(f"Formatted {len(dataset)} examples")

    # Show example
    if len(dataset) > 0:
        print("\n--- Example formatted text (first 500 chars) ---")
        print(dataset[0]["text"][:500])
        print("...")
        print("-" * 50)

    # Setup trainer
    print("\nSetting up trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            save_steps=args.save_steps,
            save_total_limit=3,
            logging_steps=args.logging_steps,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=args.seed,
            max_length=args.max_seq_length,
            dataset_text_field="text",
            packing=False,
            report_to="wandb",
            run_name=args.wandb_run_name,
            dataset_num_proc=1,  # Disable multiprocessing to avoid pickle errors with Unsloth
            dataloader_num_workers=0,  # Single process data loading
        ),
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
