import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType
import json
from datasets import load_dataset
from typing import Dict, List

class TinyAgentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        # Load and process the data
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert dictionary to list of examples
        self.examples = []
        for key, value in raw_data.items():
            # We only need the input and raw_output from each example
            for output_item in value['output']:
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
        
        # Format input and output
        input_text = f"Input: {item['input']}\n"
        output_text = f"Output: {item['raw_output']}"
        full_text = input_text + output_text
        
        input_length = len(self.tokenizer(input_text)['input_ids'])
        output_length = len(self.tokenizer(output_text)['input_ids'])
        # Tokenize input and full text
        input_encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        full_encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels: -100 for input tokens (they won't contribute to loss),
        # actual token ids for output tokens
        labels = full_encodings['input_ids'].clone().unsqueeze(0)
        # input_length = input_encodings['input_ids'].shape[1]
        labels[:input_length] = -100  # Mask out the input tokens in loss calculation

        return {
            'input_ids': full_encodings['input_ids'],
            'attention_mask': full_encodings['attention_mask'],
            'labels': labels
        }

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    labels = [[l for l in label if l != -100] for label in labels]
    predictions = [[p for p, l in zip(pred, label) if l != -100] 
                  for pred, label in zip(predictions, labels)]
    return {}

def freeze_base_model(model):
    """Freeze all parameters of the base model"""
    for param in model.parameters():
        param.requires_grad = False
    return model

def train():
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model and tokenizer initialization
    model_name = "squeeze-ai-lab/TinyAgent-1.1B"
    tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with GPU support
    model = AutoModelForCausalLM.from_pretrained(  # Changed from AutoModel to AutoModelForCausalLM
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Freeze the base model parameters
    model = freeze_base_model(model)

    # PEFT Configuration
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,  # Changed from QUESTION_ANS to CAUSAL_LM
        num_virtual_tokens=128,
        token_dim=model.config.hidden_size,
        num_transformer_submodules=1,
        prefix_projection=False,
        inference_mode=False,
        num_attention_heads=model.config.num_attention_heads,
        num_layers=model.config.num_hidden_layers
    )

    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset preparation
    train_dataset = TinyAgentDataset("../tinyagent_dataset/training_data.json", tokenizer)
    eval_dataset = TinyAgentDataset("../tinyagent_dataset/testing_data.json", tokenizer)

    outputs = model.generate(input_ids=train_dataset[0]["input_ids"].to('cuda:0'), attention_mask=train_dataset[0]["attention_mask"].to('cuda:0'), max_length=512, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Training arguments with GPU settings
    training_args = TrainingArguments(
        output_dir="prefix_tuning_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir="logs",
        logging_steps=100,
        learning_rate=1e-3,
        warmup_steps=100,
        save_steps=1000,
        load_best_model_at_end=True,
        fp16=True,
        remove_unused_columns=False,
        no_cuda=False,
        dataloader_num_workers=4,
        # Removed fp16_opt_level as it's not needed with new Accelerator
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Save the model
    model.save_pretrained("final_prefix_tuned_model")
    tokenizer.save_pretrained("final_prefix_tuned_model")

def inference(input_text):
    # Load the trained model with GPU support
    model_name = "final_prefix_tuned_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Tokenize input
    inputs = tokenizer(f"Input: {input_text}\nOutput:", 
                      return_tensors="pt", 
                      padding=True)
    
    # Move inputs to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode and process output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = generated_text.split("Output:")[-1].strip()
    
    return output_text

if __name__ == "__main__":
    # Train the model
    train()
    
    # Example inference
    test_input = "Tell me about the weather"
    result = inference(test_input)
    print(f"Input: {test_input}")
    print(f"Output: {result}")