import json

def extract_top_n_examples(input_file, output_file, n=10):
    # JSON 파일 읽기
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 처음 n개의 항목 선택
    first_n_items = dict(list(data.items())[:n])
    
    # 새 파일에 예쁘게 저장 (indent=2로 가독성 있게 포맷팅)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(first_n_items, f, indent=2, ensure_ascii=False)
    
    print(f"성공적으로 처음 {n}개의 예제를 {output_file}에 저장했습니다.")
    print(f"추출된 예제 수: {len(first_n_items)}")

if __name__ == "__main__":
    input_file = 'tinyagent_dataset/simple_test_data.json'
    output_file = 'tinyagent_dataset/simple_test_10_examples.json'
    extract_top_n_examples(input_file, output_file)