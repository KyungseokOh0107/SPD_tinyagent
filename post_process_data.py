import json
import ast
def transform_training_data(input_file, output_file):
    """
    training_data.json을 읽어서 단순화된 형태의 simple_train_data.json으로 변환하는 함수
    
    Args:
        input_file (str): 입력 파일 경로 (training_data.json)
        output_file (str): 출력 파일 경로 (simple_train_data.json)
    """
    try:
        # 입력 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # 변환된 데이터를 저장할 딕셔너리
        simple_data = {}
        
        # 각 uuid에 대해 데이터 변환
        for uuid, data in training_data.items():
            # 기본 input 가져오기
            input_text = data.get('input', '')
            
            # output 리스트에서 필요한 정보 추출
            output_list = data.get('output', [])
            system_content = ''
            raw_output = ''
            
            # output 리스트를 순회하며 필요한 정보 추출
            for output_item in output_list:
                if output_item.get('type') == 'plan':
                    raw_input_str = output_item.get('raw_input', {})
                    try:
                        # 문자열을 Python 리스트/딕셔너리로 안전하게 변환
                        raw_input_list = ast.literal_eval(raw_input_str)
                        
                        # system content 추출
                        for item in raw_input_list:
                            if item.get('role') == 'system':
                                system_content = item.get('content', '')
                                break
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: UUID {uuid}의 raw_input 파싱 중 오류 발생 - {str(e)}")
                        continue

                    raw_output = output_item.get('raw_output', '')
                    break  # 첫 번째 'plan' 타입만 처리
            
            # 변환된 형식으로 저장
            simple_data[uuid] = {
                'system': system_content,
                'input': input_text,
                'raw_input' : raw_input_str,
                'output': raw_output
            }
        
        # 결과를 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simple_data, f, ensure_ascii=False, indent=2)
            
        print(f"변환 완료: {output_file} 파일이 생성되었습니다.")
        
    except FileNotFoundError:
        print(f"Error: {input_file} 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print(f"Error: {input_file} 파일의 JSON 형식이 올바르지 않습니다.")
    except Exception as e:
        print(f"Error: 처리 중 오류가 발생했습니다 - {str(e)}")

# 사용 예시
if __name__ == "__main__":
    transform_training_data('tinyagent_dataset/testing_data.json', 'tinyagent_dataset/simple_test_data.json')
    transform_training_data('tinyagent_dataset/training_data.json', 'tinyagent_dataset/simple_train_data.json')