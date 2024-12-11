import json
import requests

def extract_inputs_from_json(json_file_path):
    """
    Extracts all inputs from the JSON file containing test data.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        list: List of input strings from each example
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    inputs = []
    for key, example in data.items():
        # Extract the input from each example
        if 'input' in example:
            inputs.append(example['input'])
    
    return inputs

def run_latency_test() :
    file_path = "tinyagent_dataset/simple_test_data.json"
    inputs = extract_inputs_from_json(file_path)
    print("List of inputs:")
    
    num = 20
    for i in range(num):
        if i >= len(inputs):
            break
        input_str = inputs[i]
        print(f"trial : {i}")
        query = input_str
        response = requests.post('http://127.0.0.1:50002/generate', json={'query': query})
    pass

# Example usage:
if __name__ == "__main__":
    run_latency_test()