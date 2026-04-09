"""Scaffold LoRA fine-tuning script using Transformers + PEFT.

This is a scaffold. Run in a GPU cloud instance with CUDA available.
Follow README_LLM.md for step-by-step instructions.
"""
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = os.getenv('BASE_MODEL', 'mistralai/Mistral-7B-Instruct-v0')
DATA_PATH = os.getenv('TRAIN_DATA', 'llm_synthetic.jsonl')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'lora_finetuned')

def load_data(path):
    return load_dataset('json', data_files=path)

def main():
    ds = load_data(DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map='auto', trust_remote_code=True)

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj','v_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )

    model = get_peft_model(model, lora_config)

    def tokenize_fn(ex):
        prompts = []
        for h,a in zip(ex['history_item'], ex['assistant_response']):
            prompt = f"Context: {json.dumps(h)}\nAssistant:"
            prompts.append(prompt + a)
        return tokenizer(prompts, truncation=True, padding='max_length', max_length=1024)

    tokenized = ds.map(lambda x: tokenizer(x['assistant_response'], truncation=True, padding='max_length', max_length=1024), batched=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        output_dir=OUTPUT_DIR,
        logging_steps=10,
        save_total_limit=3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'] if 'train' in tokenized else tokenized,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)

if __name__ == '__main__':
    main()
