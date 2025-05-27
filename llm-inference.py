#!/usr/bin/env python3
"""
Minimal vLLM inference script with GPU optimization best practices
"""

import json
import argparse
from vllm import LLM, SamplingParams

def load_prompts(json_path: str) -> list[str]:
    """Load prompts from JSON file with system_prompt + user_prompts structure"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle your specific JSON structure
    if isinstance(data, dict) and 'system_prompt' in data and 'user_prompts' in data:
        system_prompt = data['system_prompt']
        user_prompts = data['user_prompts']
        
        # Combine system prompt with each user prompt
        combined_prompts = []
        for user_prompt in user_prompts:
            text = user_prompt.get('text', '')
            combined: str = f"{system_prompt}\n\n{text}"
            combined_prompts.append(combined)
        
        return combined_prompts
    else:
        raise ValueError("Invalid JSON structure. Expected 'system_prompt' and 'user_prompts' keys.")


def run_inference(model_name: str, prompts: list[str]) -> list[str]:
    """Run optimized batch inference"""
    
    # Determine optimal tensor parallelism based on model size
    if 'gpt2' in model_name.lower():
        tp_size = 1  # GPT-2 Medium is small, single GPU sufficient
        gpu_util = 0.7
    elif 'gemma' in model_name.lower():
        tp_size = 1  # Gemma 2B is small
        gpu_util = 0.8
    else:
        tp_size = 2  # Default for unknown models
        gpu_util = 0.85
    
    print(f"Loading {model_name} with tensor_parallel_size={tp_size}")
    
    # Initialize model with optimizations
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_util,
        max_model_len=1024,  # Reasonable context for these models
        enable_prefix_caching=True,  # Enable KV caching
        disable_log_stats=False,
        swap_space=4,  # 4GB swap space for memory efficiency
        cpu_offload_gb=0,  # Keep everything on GPU
        enforce_eager=False,  # Use CUDA graphs when possible
    )
    
    # Sampling parameters optimized for quality and speed
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000,  # Max 1k output tokens as requested
        stop_token_ids=None,
        skip_special_tokens=True,
    )
    
    print(f"Running batch inference on {len(prompts)} prompts...")
    
    # Batch inference - vLLM automatically optimizes batching
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract generated text
    results = [output.outputs[0].text for output in outputs]
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Minimal vLLM inference script')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--model', 
                       choices=['gpt2-medium', 'google/gemma-2b'], 
                       default='gpt2-medium',
                       help='Model to use for inference')
    parser.add_argument('--output', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_prompts(args.input)
    print(f"Loaded {len(prompts)} prompts")
    
    # Run inference
    results = run_inference(args.model, prompts)
    
    # Prepare output
    output_data = [
        {"prompt": prompt, "completion": result}
        for prompt, result in zip(prompts, results)
    ]
    
    # Save or print results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output_data, indent=2))

if __name__ == "__main__":
    main()