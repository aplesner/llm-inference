#!/usr/bin/env python3
"""
HuggingFace inference script using datasets.map() for optimized batching
"""

import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset

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
            combined = f"{system_prompt}\n\n{text}"
            combined_prompts.append(combined)
        
        return combined_prompts
    
    # Fallback to original logic for other formats
    elif isinstance(data, list):
        if isinstance(data[0], dict):
            return [item.get('prompt', item.get('text', next(iter(item.values())))) for item in data]
        else:
            return [str(item) for item in data]
    elif isinstance(data, dict):
        return list(data.values()) if 'prompts' not in data else data['prompts']
    
    return [str(data)]

def setup_pipeline(model_name: str):
    """Setup HuggingFace pipeline with optimizations"""
    
    device_count = torch.cuda.device_count()
    
    # Determine device and dtype
    if device_count == 0:
        device = -1  # CPU
        torch_dtype = torch.float32
    else:
        device = 0  # Primary GPU, pipeline handles multi-GPU
        torch_dtype = torch.float16
    
    print(f"Using {device_count} GPU(s), dtype: {torch_dtype}")
    
    # Load tokenizer separately to set padding correctly
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='left',  # Fix the padding warning
        truncation_side='left'
    )
    
    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model separately to resize embeddings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device_count > 1 else None,
        trust_remote_code=True,
        use_cache=True,
        low_cpu_mem_usage=True,
    )
    
    # Fix the CUDA indexing issue by resizing token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Create pipeline with pre-configured model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device if device_count <= 1 else None,  # Let device_map handle multi-GPU
    )
    
    return pipe

def generate_batch(examples, pipe):
    """Batch generation function for dataset.map()"""
    
    # Generate for batch of prompts with better error handling
    try:
        outputs = pipe(
            examples['prompt'],
            max_new_tokens=1000,
            max_length=None,  # Don't set max_length, use max_new_tokens instead
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=pipe.tokenizer.pad_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,  # Only return generated text
            batch_size=4,  # Smaller batch size to avoid memory issues
            truncation=True,  # Enable truncation
        )
        
        # Extract generated text
        if isinstance(outputs[0], list):
            # Handle case where output is list of lists
            generated_texts = [output[0]['generated_text'] for output in outputs]
        else:
            # Handle case where output is list of dicts
            generated_texts = [output['generated_text'] for output in outputs]
            
    except Exception as e:
        print(f"Error in batch generation: {e}")
        # Fallback: process one by one
        generated_texts = []
        for prompt in examples['prompt']:
            try:
                output = pipe(
                    prompt,
                    max_new_tokens=1000,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=pipe.tokenizer.pad_token_id,
                    eos_token_id=pipe.tokenizer.eos_token_id,
                    return_full_text=False,
                    truncation=True,
                )
                generated_texts.append(output[0]['generated_text'])
            except Exception as inner_e:
                print(f"Error generating for single prompt: {inner_e}")
                generated_texts.append("[GENERATION_ERROR]")
    
    return {'completion': generated_texts}

def run_inference(model_name: str, prompts: list[str]) -> list[str]:
    """Run optimized inference using HuggingFace datasets and pipeline"""
    
    print(f"Setting up pipeline for {model_name}...")
    pipe = setup_pipeline(model_name)
    
    # Create dataset from prompts
    dataset = Dataset.from_dict({'prompt': prompts})
    
    # Now we can use larger batch sizes since embeddings are properly sized
    if 'gpt2' in model_name.lower():
        batch_size = 8  # GPT-2 is small
    elif 'gemma' in model_name.lower():
        batch_size = 4  # Gemma 2B moderate
    else:
        batch_size = 2  # Conservative default
    
    print(f"Running dataset.map() with batch_size={batch_size}")
    
    # Use dataset.map() for optimized batching
    results_dataset = dataset.map(
        lambda examples: generate_batch(examples, pipe),
        batched=True,
        batch_size=batch_size,
        remove_columns=['prompt'],  # Keep only completions
        desc="Generating completions"
    )
    
    return results_dataset['completion']

def main():
    parser = argparse.ArgumentParser(description='HuggingFace inference with datasets.map()')
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
