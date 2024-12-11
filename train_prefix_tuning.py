import json
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PrefixTuningConfig, get_peft_model
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Prefix-tuning for TinyAgent')
    parser.add_argument('--num_virtual_tokens', type=int, default=128,
                        help='Number of virtual tokens for prefix-tuning')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--train_data', type=str, 
                        default='/home/kyoungseokoh/SPD_tinyagent/tinyagent_dataset/simple_train_data.json',
                        help='Path to training data')
    parser.add_argument('--test_data', type=str,
                        default='/home/kyoungseokoh/SPD_tinyagent/tinyagent_dataset/simple_test_data.json',
                        help='Path to test data')
    parser.add_argument('--save_path', type=str, default='./prefix_tuning_models',
                        help='Path to save the models')
    parser.add_argument('--output_file', type=str, default='test_results.txt',
                        help='Path to save test results')
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = list(self.data.values())[idx]
        # input에서 system을 제외하고 토크나이징
        input_text = item['input'].replace(item['system'], '').strip()
        
        # 입력과 출력 토크나이징
        input_encoding = self.tokenizer(input_text, 
                                      max_length=self.max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')
        
        output_encoding = self.tokenizer(item['output'],
                                       max_length=self.max_length,
                                       padding='max_length',
                                       truncation=True,
                                       return_tensors='pt')

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': output_encoding['input_ids'].squeeze()
        }

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_peft_config(model, num_virtual_tokens):
    peft_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=num_virtual_tokens,
        token_dim=model.config.hidden_size,
        num_transformer_submodules=None,  # 모든 layer에 적용
        prefix_projection=True,
        inference_mode=False,
    )
    return peft_config

def train(model, train_loader, val_loader, device, save_path, learning_rate):
    # AdamW optimizer 사용 (가장 일반적으로 사용되는 optimizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    
    num_training_steps = len(train_loader)
    validation_steps = num_training_steps // 9  # train set의 1/9 마다 validation
    
    model.train()
    step = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        step += 1
        
        # Validation 체크
        if step % validation_steps == 0:
            val_loss = validate(model, val_loader, device)
            print(f"Step: {step}/{num_training_steps}, Validation Loss: {val_loss}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 기존 모델 삭제
                for file in os.listdir(save_path):
                    if file.startswith('best_model'):
                        os.remove(os.path.join(save_path, file))
                # 새로운 모델 저장
                model.save_pretrained(os.path.join(save_path, f'best_model_{step}'))
                print(f"New best model saved with validation loss: {val_loss}")
            
            model.train()

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(val_loader)

def test_model(model, test_dataset, tokenizer, device, output_file):
    model.eval()
    results = []
    
    with torch.no_grad():
        for item in tqdm(test_dataset.values(), desc="Testing"):
            input_text = item['input'].replace(item['system'], '').strip()
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=512,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'input': input_text,
                'expected': item['output'],
                'generated': generated_text
            })
    
    # 결과를 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"Input: {result['input']}\n")
            f.write(f"Expected: {result['expected']}\n")
            f.write(f"Generated: {result['generated']}\n")
            f.write("-" * 50 + "\n")

def main():
    args = parse_args()
    
    # 저장 경로 생성
    os.makedirs(args.save_path, exist_ok=True)
    
    # 모델과 토크나이저 로드
    model_name = "squeeze-ai-lab/TinyAgent-1.1B"
    tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # PEFT 설정 및 모델 준비
    peft_config = create_peft_config(model, args.num_virtual_tokens)
    model = get_peft_model(model, peft_config)
    
    # 데이터 로드
    train_data = load_dataset(args.train_data)
    test_data = load_dataset(args.test_data)
    
    # Dataset 생성
    full_dataset = CustomDataset(train_data, tokenizer)
    
    # Train/Val 분할 (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("Starting training with following parameters:")
    print(f"Number of virtual tokens: {args.num_virtual_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {device}")
    
    # 학습
    train(model, train_loader, val_loader, device, args.save_path, args.learning_rate)
    
    # 최종 모델로 테스트
    best_model_path = max(
        [f for f in os.listdir(args.save_path) if f.startswith('best_model')],
        key=lambda x: int(x.split('_')[-1])
    )
    
    model = PeftModel.from_pretrained(model, os.path.join(args.save_path, best_model_path))
    test_model(model, test_data, tokenizer, device, args.output_file)

if __name__ == "__main__":
    main()