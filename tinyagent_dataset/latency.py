import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd

def create_input_with_token_length(tokenizer, target_length):
    """Create input with exact number of tokens."""
    # Create a tensor of repeated tokens (e.g., using pad token)
    # We use pad token since it's a neutral token that won't affect the model's behavior significantly
    input_ids = torch.full((1, target_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.ones((1, target_length), dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def measure_latency(model, inputs, target_output_length, num_trials=10):
    """Measure prefill and decoding latency using CUDA events."""
    model.eval()
    
    # Move inputs to GPU
    inputs = {k: v.cuda() for k, v in inputs.items()}
    input_length = inputs['input_ids'].size(1)  # Get actual input length in tokens
    
    # Create CUDA events for prefill
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    
    # Create CUDA events for decoding
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)
    
    # Warm-up run
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=target_output_length,
            min_new_tokens=target_output_length,
            pad_token_id=model.config.pad_token_id
        )
    
    prefill_latencies = []
    decode_latencies = []
    
    for _ in range(num_trials):
        torch.cuda.synchronize()
        
        # Measure prefill (forward pass through input sequence)
        prefill_start.record()
        with torch.no_grad():
            outputs = model(**inputs)
        prefill_end.record()
        torch.cuda.synchronize()
        prefill_latency = prefill_start.elapsed_time(prefill_end)
        prefill_latencies.append(prefill_latency)
        
        # Measure decoding (generation)
        decode_start.record()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=target_output_length,
                min_new_tokens=target_output_length,
                pad_token_id=model.config.pad_token_id
            )
        decode_end.record()
        torch.cuda.synchronize()
        decode_latency = decode_start.elapsed_time(decode_end)
        decode_latencies.append(decode_latency)
    
    return {
        'input_length': input_length,
        'prefill': {
            'mean': np.mean(prefill_latencies),
            'std': np.std(prefill_latencies)
        },
        'decode': {
            'mean': np.mean(decode_latencies),
            'std': np.std(decode_latencies)
        }
    }

def analyze_results(results):
    """Analyze latency results and calculate speedup."""
    analysis = []
    
    for dataset in results:
        for input_type in ['full_input', 'input_only']:
            data = results[dataset][input_type]
            
            # Calculate speedup compared to full_input case
            if input_type == 'input_only':
                full_input_data = results[dataset]['full_input']
                prefill_speedup = full_input_data['prefill']['mean'] / data['prefill']['mean']
                decode_speedup = full_input_data['decode']['mean'] / data['decode']['mean']
            else:
                prefill_speedup = 1.0
                decode_speedup = 1.0
            
            analysis.append({
                'Dataset': dataset,
                'Input Type': input_type,
                'Input Length (tokens)': data['input_length'],
                'Output Length (tokens)': data['output_length'],
                'Prefill Latency (ms)': f"{data['prefill']['mean']:.2f} ± {data['prefill']['std']:.2f}",
                'Decode Latency (ms)': f"{data['decode']['mean']:.2f} ± {data['decode']['std']:.2f}",
                'Prefill Speedup': f"{prefill_speedup:.2f}x",
                'Decode Speedup': f"{decode_speedup:.2f}x"
            })
    
    return pd.DataFrame(analysis)

def main():
    # Model initialization
    model_name = "squeeze-ai-lab/TinyAgent-1.1B"
    tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test cases configuration
    test_cases = {
        'test_data': {
            'full_input_length': 1757,     # system_prompt(1716) + input(41)
            'input_only_length': 41,
            'output_length': 87
        },
        'train_data': {
            'full_input_length': 3403,     # system_prompt(3362) + input(41)
            'input_only_length': 41,
            'output_length': 88
        }
    }
    
    # Run measurements
    results = {}
    
    for dataset_name, config in test_cases.items():
        results[dataset_name] = {}
        
        # Case 1: full input (system_prompt + input)
        full_inputs = create_input_with_token_length(tokenizer, config['full_input_length'])
        results[dataset_name]['full_input'] = measure_latency(
            model, 
            full_inputs, 
            config['output_length']
        )
        results[dataset_name]['full_input']['output_length'] = config['output_length']
        
        # Case 2: input only
        input_only = create_input_with_token_length(tokenizer, config['input_only_length'])
        results[dataset_name]['input_only'] = measure_latency(
            model, 
            input_only, 
            config['output_length']
        )
        results[dataset_name]['input_only']['output_length'] = config['output_length']
    
    # Analyze and display results
    analysis_df = analyze_results(results)
    
    print("\nTinyLlama Latency Analysis:")
    print("=" * 120)
    print(analysis_df.to_string(index=False))
    
    # Save results to CSV
    analysis_df.to_csv('tinyllama_latency_analysis.csv', index=False)
    print("\nResults have been saved to 'tinyllama_latency_analysis.csv'")

if __name__ == "__main__":
    main()