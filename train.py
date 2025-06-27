import argparse, json, os, math
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = f"# Python 3\n# {obj['prompt'].strip()}\n{obj['completion'].rstrip()}\n"
            records.append({"text": text})
    return Dataset.from_list(records)

def main(args):
    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    print("Adding LoRA adapters …")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    print("Loading dataset …")
    dataset = load_jsonl(args.train_file)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    print("Tokenization complete.")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=args.num_epochs,
        learning_rate=5e-4,
        fp16=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    print("Starting training …")
    trainer.train()
    print("Saving final model …")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--model_name", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("--output_dir", type=str, default="model")
    parser.add_argument("--num_epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)