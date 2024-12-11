from transformers import AutoTokenizer, LlamaTokenizerFast
import json

def check_tokenizer_info(model_id):
    """
    모델의 tokenizer 정보를 확인하는 함수
    
    Args:
        model_id (str): Hugging Face 모델 ID
    """
    try:
        breakpoint()
        # tokenizer 로드
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # load tokenizer named "LlamaTokenizerFast"
        
        tokenizer =  LlamaTokenizerFast.from_pretrained(model_id)
        model_id = "squeeze-ai-lab/TinyAgent-1.1B"
        tokenizer_2 = LlamaTokenizerFast.from_pretrained(model_id)
        breakpoint()
        # tokenizer의 기본 정보 출력
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        # tokenizer config 정보 확인
        config = tokenizer.init_kwargs
        print("\nTokenizer configuration:")
        print(json.dumps(config, indent=2))
        
        # 기본 토큰 확인
        special_tokens = {
            "PAD token": tokenizer.pad_token,
            "UNK token": tokenizer.unk_token,
            "BOS/SOS token": tokenizer.bos_token,
            "EOS token": tokenizer.eos_token,
            "SEP token": tokenizer.sep_token if hasattr(tokenizer, 'sep_token') else None,
            "CLS token": tokenizer.cls_token if hasattr(tokenizer, 'cls_token') else None,
            "MASK token": tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else None
        }
        
        print("\nSpecial tokens:")
        for token_name, token in special_tokens.items():
            print(f"{token_name}: {token}")
            
        # 간단한 테스트 문장으로 토크나이징 테스트
        test_text = "Hello, how are you?"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        print(f"\nTest encoding/decoding:")
        print(f"Original text: {test_text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

# 모델 확인
model_id = "squeeze-ai-lab/TinyAgent-1.1B"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
check_tokenizer_info(model_id)