from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os

def find_model_files(directory, target_files=['config.json', 'tokenizer.json']):
    """Recursively search for required model files"""
    for root, dirs, files in os.walk(directory):
        if any(target in files for target in target_files):
            print(f"Found model files at: {root}")
            print(f"Files: {files}")
            return root
    return None

def check_model_structure():
    """Check model directory structure"""
    print("Checking model directory structure...")
    
    qwen_path = "./Qwen2.5-0.5B-Instruct/Qwen"
    if os.path.exists(qwen_path):
        print("Qwen directory contents:", os.listdir(qwen_path))
        
        for item in os.listdir(qwen_path):
            item_path = os.path.join(qwen_path, item)
            if os.path.isdir(item_path):
                print(f"Subdirectory {item} contents:", os.listdir(item_path))
    else:
        print("WARNING: Qwen directory not found")
    
    model_path = find_model_files("./Qwen2.5-0.5B-Instruct")
    return model_path

def test_inference():
    """Test model inference capability"""
    print("\nTesting model inference...")
    
    model_name = "./Qwen2.5-0.5B-Instruct/Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        prompt = "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"
        messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        print(f"Input: {prompt}")
        print(f"Token shape: {model_inputs.input_ids.shape}")
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Model response:\n{response}")
        return True
        
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

def check_dataset():
    """Check dataset loading"""
    print("\nChecking dataset loading...")
    
    try:
        dataset = load_dataset('openai/gsm8k', 'main')
        print("Online dataset loaded successfully!")
    except:
        try:
            dataset = load_dataset('json', data_files={
                'train': './data/gsm8k/train.jsonl', 
                'test': './data/gsm8k/test.jsonl'
            })
            print("Local dataset loaded successfully!")
        except Exception as e:
            print(f"Dataset loading failed: {e}")
            return None
    
    print(f"Training set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")
    
    sample = dataset['train'][0]
    print(f"\nSample data:")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    
    return dataset

if __name__ == "__main__":
    print("Starting environment check...\n")
    
    model_ok = check_model_structure()
    inference_ok = test_inference()
    dataset_ok = check_dataset()
    
    print("\n" + "="*50)
    if model_ok and inference_ok and dataset_ok:
        print("All checks passed - ready for training")
    else:
        print("Issues detected - check setup")
    print("="*50)