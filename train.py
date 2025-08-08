import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import swanlab

# System prompt template
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML tags"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    """Extract ground truth answer from GSM8K format"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def prepare_dataset(split="train") -> Dataset:
    """Load and format GSM8K dataset"""
    try:
        dataset = load_dataset('openai/gsm8k', 'main')
        print("Online dataset loaded successfully!")
    except:
        dataset = load_dataset('json', data_files={
            'train': './data/gsm8k/train.jsonl', 
            'test': './data/gsm8k/test.jsonl'
        })
        print("Local dataset loaded successfully!")
    
    data = dataset[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

# Reward functions for GRPO training
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Answer correctness reward (2.0 points)"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", 
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """Integer answer format reward (0.5 points)"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Strict XML format compliance reward (0.5 points)"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Lenient XML format compliance reward (0.5 points)"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """XML tag quality scoring function"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """XML format scoring calculation"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def main():
    # Initialize experiment tracking
    swanlab.login(api_key="your_api_key")
    swanlab.init(
        project="GRPO-train",
        experiment_name="qwen-grpo-gsm8k",
        description="GRPO training on Qwen2.5-0.5B with GSM8K dataset",
        config={
            "model": "Qwen2.5-0.5B-Instruct", 
            "dataset": "GSM8K",
            "method": "GRPO"
        }
    )
    
    # Training configuration
    model_name = "./Qwen2.5-0.5B-Instruct/Qwen/Qwen2.5-0.5B-Instruct"
    output_dir = "outputs/Qwen2.5-0.5B-reasoning-GRPO"
    run_name = "Qwen2.5-0.5B-GRPO-gsm8k"
    
    # Load dataset
    dataset = prepare_dataset("train")
    print(f"Training dataset size: {len(dataset)}")
    
    # GRPO training arguments
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        vllm_gpu_memory_utilization=0.3,
        vllm_device="cuda:0",
        report_to="swanlab" 
    )
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    # Start training
    print("Starting GRPO training...")
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    print(f"Training completed. Model saved to: {output_dir}")

if __name__ == "__main__":
    main()