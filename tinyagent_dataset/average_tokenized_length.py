from transformers import AutoTokenizer
import json
import statistics
from typing import Dict, List
import os

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def analyze_token_lengths(data: Dict, dataset_name: str) -> Dict:
    """Analyze token lengths for a single dataset"""
    # Lists to store lengths
    system_lengths = []
    input_lengths = []
    output_lengths = []
    
    # Process each example in the data
    for key, example in data.items():
        # Get token lengths
        system_length = len(tokenizer.encode(example['system']))
        input_length = len(tokenizer.encode(example['input']))
        output_length = len(tokenizer.encode(example['output']))
        
        # Store lengths
        system_lengths.append(system_length)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
    
    # Calculate averages
    avg_system = statistics.mean(system_lengths)
    avg_input = statistics.mean(input_lengths)
    avg_output = statistics.mean(output_lengths)
    
    # Calculate min/max values
    min_system = min(system_lengths)
    max_system = max(system_lengths)
    min_input = min(input_lengths)
    max_input = max(input_lengths)
    min_output = min(output_lengths)
    max_output = max(output_lengths)
    
    # Print results for this dataset
    print(f"\n=== Analysis for {dataset_name} ===")
    print(f"Number of examples: {len(data)}")
    print("\nSystem prompt tokens:")
    print(f"  Average: {avg_system:.2f}")
    print(f"  Min: {min_system}")
    print(f"  Max: {max_system}")
    
    print("\nInput tokens:")
    print(f"  Average: {avg_input:.2f}")
    print(f"  Min: {min_input}")
    print(f"  Max: {max_input}")
    
    print("\nOutput tokens:")
    print(f"  Average: {avg_output:.2f}")
    print(f"  Min: {min_output}")
    print(f"  Max: {max_output}")
    
    return {
        'system_lengths': system_lengths,
        'input_lengths': input_lengths,
        'output_lengths': output_lengths,
        'averages': {
            'system': avg_system,
            'input': avg_input,
            'output': avg_output
        },
        'min_max': {
            'system': (min_system, max_system),
            'input': (min_input, max_input),
            'output': (min_output, max_output)
        }
    }

def analyze_multiple_datasets(file_paths: List[str]):
    """Analyze multiple datasets and show results"""
    results = {}
    
    for file_path in file_paths:
        # Get dataset name from file path
        dataset_name = os.path.basename(file_path).replace('.json', '')
        
        # Load the JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Analyze this dataset
        results[dataset_name] = analyze_token_lengths(data, dataset_name)
    
    return results

if __name__ == "__main__":
    # Define file paths
    file_paths = [
        '/home/kyoungseokoh/SPD_tinyagent/tinyagent_dataset/simple_test_data.json',
        '/home/kyoungseokoh/SPD_tinyagent/tinyagent_dataset/simple_train_data.json'
    ]
    
    # Run analysis
    results = analyze_multiple_datasets(file_paths)