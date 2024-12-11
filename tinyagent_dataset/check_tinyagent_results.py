import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_model_and_tokenizer():
    # Load model and tokenizer
    model_name = "squeeze-ai-lab/TinyAgent-1.1B"
    tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # GPU 사용 가능시 GPU로 이동
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, tokenizer, device

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_response(model, tokenizer, input_text, device):
    # 입력 텍스트 토큰화
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    
    # 모델 출력 생성
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids,num_return_sequences=1,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,min_new_tokens=18,max_new_tokens=428)
    breakpoint()
    # 출력 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 입력 텍스트 제거하여 실제 생성된 응답만 반환
    input_length = inputs.input_ids.size(1)
    actual_response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return actual_response

def evaluate_dataset():
    # 모델과 토크나이저 로드
    model, tokenizer, device = load_model_and_tokenizer()
    
    # 데이터셋 로드
    dataset_path = "/home/kyoungseokoh/SPD_tinyagent/tinyagent_dataset/simple_test_10_examples.json"
    dataset = load_dataset(dataset_path)
    
    # 결과 비교를 위한 리스트
    mismatched_examples = []
    
    # 각 예제에 대해 평가 수행
    for example_id, example in tqdm(dataset.items()):
        # 모델 출력 생성
        model_output = generate_response(model, tokenizer, example['raw_input'], device)
        
        breakpoint()
        # 결과가 일치하지 않는 경우 저장
        if model_output.strip() != example['output'].strip():
            mismatched_examples.append({
                'example_id': example_id,
                'raw_input': example['raw_input'],
                'expected_output': example['output'],
                'model_output': model_output
            })
    
    # 결과 출력
    print(f"\n총 예제 수: {len(dataset)}")
    print(f"불일치하는 예제 수: {len(mismatched_examples)}")
    
    if mismatched_examples:
        print("\n불일치하는 예제들:")
        for idx, example in enumerate(mismatched_examples, 1):
            print(f"\n{idx}. Example ID: {example['example_id']}")
            print(f"입력: {example['raw_input']}")
            print(f"기대 출력: {example['expected_output']}")
            print(f"모델 출력: {example['model_output']}")

if __name__ == "__main__":
    evaluate_dataset()