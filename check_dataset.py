from datasets import load_dataset
import pandas as pd
import os
from huggingface_hub import snapshot_download
import json

def load_and_print_tinyagent():
    try:
        # 먼저 데이터셋을 로컬에 다운로드
        local_dir = snapshot_download(
            repo_id="squeeze-ai-lab/TinyAgent-dataset",
            repo_type="dataset",
            local_dir="./tinyagent_dataset"
        )

        breakpoint()
        
        # 데이터셋의 100줄 출력
        with open(os.path.join(local_dir, 'testing_data.json'), 'r') as f:
            testing_data = json.load(f)
            df = pd.DataFrame(testing_data)
            print("=== Testing Data ===")
            print(df.head(100))

        breakpoint()
        # 로컬에서 데이터셋 로드
        dataset = load_dataset('json',data_files={'train': os.path.join(local_dir, 'training_data.json'),'test': os.path.join(local_dir, 'testing_data.json')})
        
        # 데이터셋 구조 출력
        print("=== Dataset Structure ===")
        print(dataset)
        print("\n=== Dataset Features ===")
        print(dataset['train'].features)
        
        # 처음 5개 샘플 출력
        print("\n=== First 5 Examples ===")
        for i, example in enumerate(dataset['train'].select(range(5))):
            print(f"\nExample {i+1}:")
            print("-" * 50)
            print("Input:")
            print(example['input'])
            print("\nOutput:")
            print(example['output'])
            print("-" * 50)
            
        # 데이터셋 기본 통계
        print("\n=== Dataset Statistics ===")
        print(f"Training set size: {len(dataset['train'])}")
        print(f"Validation set size: {len(dataset['validation'])}")
        print(f"Test set size: {len(dataset['test'])}")
        
        # 입력과 출력 길이 통계
        input_lengths = [len(x['input'].split()) for x in dataset['train']]
        output_lengths = [len(x['output'].split()) for x in dataset['train']]
        
        print("\n=== Length Statistics ===")
        print(f"Input length (words) - Average: {sum(input_lengths)/len(input_lengths):.1f}, "
              f"Min: {min(input_lengths)}, Max: {max(input_lengths)}")
        print(f"Output length (words) - Average: {sum(output_lengths)/len(output_lengths):.1f}, "
              f"Min: {min(output_lengths)}, Max: {max(output_lengths)}")
        
        return dataset

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("\nTrying alternative loading method...")
        
        try:
            # 대체 방법: 직접 JSON 파일 다운로드
            dataset = load_dataset(
                "squeeze-ai-lab/TinyAgent-dataset",
                use_auth_token=True  # Hugging Face 토큰이 필요할 수 있습니다
            )
            print("Successfully loaded dataset using alternative method!")
            return dataset
            
        except Exception as e2:
            print(f"Alternative loading method also failed: {str(e2)}")
            print("\nPlease make sure you:")
            print("1. Have a stable internet connection")
            print("2. Have logged in to Hugging Face (`huggingface-cli login`)")
            print("3. Have sufficient disk space")
            return None

if __name__ == "__main__":
    dataset = load_and_print_tinyagent()