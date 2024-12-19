import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType
import json
from datasets import load_dataset
from typing import Dict, List
import wandb
import os
from datetime import datetime
import argparse

class TinyAgentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.examples = []
        for idx, (key, value) in enumerate(raw_data.items()):
            output_item = value['output'][0]
            if 'raw_output' in output_item:
                self.examples.append({
                    'input': value['input'],
                    'raw_output': output_item['raw_output']
                })
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        
        item = self.examples[idx]
        
        input_text = f"Input: {item['input']}\n"
        output_text = f"Output: {item['raw_output']}"
        full_text = input_text + output_text
        
        input_ids_input = self.tokenizer(input_text, add_special_tokens=True)['input_ids']
        input_ids_full = self.tokenizer(full_text, add_special_tokens=True)['input_ids']
        
        input_length = len(input_ids_input)
        full_length = len(input_ids_full)
        
        full_encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        labels = full_encodings['input_ids'].clone()

        pad_length = self.max_length - full_length
        if pad_length > 0:
            labels[0, :pad_length] = -100
            labels[0, pad_length:pad_length + input_length] = -100
        else:
            labels[0, :input_length] = -100
            
        return {
            'input_ids': full_encodings['input_ids'].squeeze(),
            'attention_mask': full_encodings['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    labels = [[l for l in label if l != -100] for label in labels]
    predictions = [[p for p, l in zip(pred, label) if l != -100] 
                  for pred, label in zip(predictions, labels)]
    
    # Add metrics for wandb logging
    metrics = {
        "eval/num_samples": len(labels),
        "eval/avg_sequence_length": sum(len(l) for l in labels) / len(labels) if labels else 0
    }
    
    # Log metrics to wandb
    wandb.log(metrics)
    
    return metrics

class WandbTrainerCallback(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_loss = float('inf')
    
    def log(self, logs: Dict[str, float]) -> None:
        # Call the original logging
        super().log(logs)
        
        # Add custom wandb logging
        if logs is not None:
            # Track training loss
            if "loss" in logs:
                wandb.log({"train/loss": logs["loss"]})
            
            # Track learning rate
            if "learning_rate" in logs:
                wandb.log({"train/learning_rate": logs["learning_rate"]})
            
            # Track best loss
            if "eval_loss" in logs:
                if logs["eval_loss"] < self.best_loss:
                    self.best_loss = logs["eval_loss"]
                wandb.log({
                    "eval/loss": logs["eval_loss"],
                    "eval/best_loss": self.best_loss
                })

def freeze_base_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train TinyAgent with prefix tuning')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default="squeeze-ai-lab/TinyAgent-1.1B",
                      help='Base model name')
    parser.add_argument('--tokenizer_name', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help='Tokenizer name')
    
    # Training configuration
    parser.add_argument('--num_virtual_tokens', type=int, default=128,
                      help='Number of virtual tokens for prefix tuning')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1,
                      help='Training batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=100,
                      help='Number of warmup steps')
    parser.add_argument('--eval_steps', type=int, default=5000,
                      help='Number of steps between evaluations')
    parser.add_argument('--save_steps', type=int, default=1000,
                      help='Number of steps between model saves')
    
    # Data paths
    parser.add_argument('--train_data_path', type=str, default="../tinyagent_dataset/training_data.json",
                      help='Path to training data')
    parser.add_argument('--eval_data_path', type=str, default="../tinyagent_dataset/testing_data.json",
                      help='Path to evaluation data')
    
    # Output configuration
    parser.add_argument('--output_dir_base', type=str, default="/home/ksoh99/SPD_tinyagent/prefixtuning/prefix_tuning_output",
                      help='Base directory for outputs')
    
    # Wandb configuration
    parser.add_argument('--wandb_project', type=str, default="tinyagent-prefix-tuning",
                      help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                      help='Wandb run name (default: auto-generated)')
    
    args = parser.parse_args()
    return args

def train(args):
    # Generate model name with parameters
    model_name_with_params = (f"tinyagent_vt{args.num_virtual_tokens}_"
                           f"lr{args.learning_rate:.0e}_"
                           f"ep{args.num_train_epochs}")
    
    # Generate run name if not provided
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{model_name_with_params}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize wandb with arguments
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)  # Convert all arguments to wandb config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wandb.log({"device": str(device)})

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    model = freeze_base_model(model)

    if args.model_name == "squeeze-ai-lab/TinyAgent-1.1B":
        model.config.key_value_dimension = 256
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.num_virtual_tokens,
            token_dim=model.config.key_value_dimension,
            num_transformer_submodules=1,
            prefix_projection=False,
            inference_mode=False,
            num_attention_heads=model.config.num_key_value_heads,
            num_layers=model.config.num_hidden_layers
        )
    else:
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.num_virtual_tokens,
            token_dim=model.config.hidden_size,
            num_transformer_submodules=1,
            prefix_projection=False,
            inference_mode=False,
            num_attention_heads=model.config.num_attention_heads,
            num_layers=model.config.num_hidden_layers
        )

    model = get_peft_model(model, peft_config)
    
    # Log model parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    wandb.log({
        "model/trainable_params": trainable_params,
        "model/total_params": total_params,
        "model/trainable_percent": (trainable_params/total_params)*100
    })
    
    model.print_trainable_parameters()

    train_dataset = TinyAgentDataset(args.train_data_path, tokenizer)
    eval_dataset = TinyAgentDataset(args.eval_data_path, tokenizer)
    
    wandb.log({
        "dataset/train_size": len(train_dataset),
        "dataset/eval_size": len(eval_dataset)
    })

    output_dir = os.path.join(args.output_dir_base, args.wandb_run_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        fp16=True,
        remove_unused_columns=False,
        no_cuda=False,
        dataloader_num_workers=4,
        report_to="wandb"
    )

    trainer = WandbTrainerCallback(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save the model with parameters in name
    final_model_path = f"final_model_{model_name_with_params}"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    wandb.log({"training_completed": True})
    wandb.finish()

def inference(input_text, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    inputs = tokenizer(f"Input: {input_text}\nOutput:", 
                      return_tensors="pt", 
                      padding=True)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_return_sequences=1,
        temperature=0,
        top_p=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = generated_text.split("Output:")[-1].strip()
    
    return output_text

if __name__ == "__main__":
    args = parse_args()
    train(args)
    
    # Example inference using the saved model with parameters in name
    model_name_with_params = (f"final_model_tinyagent_vt{args.num_virtual_tokens}_"
                           f"lr{args.learning_rate:.0e}_"
                           f"ep{args.num_train_epochs}")
    
    test_input = 'Reply to the currently selected email in Mail with the match details attached and create a new note titled "Festival Notes" to summarize the discussions.'
    result = inference(test_input, model_name_with_params)
    print(f"Input: {test_input}")
    print(f"Output: {result}")