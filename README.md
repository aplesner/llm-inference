## Usage

1. Build the Singularity container:
```bash
sudo singularity build singularity/vllm.sif singularity/vllm_inference.def
```

2. Execute the container
```bash
singularity exec --nv   --env HF_HOME=/scratch/aplesner/huggingface   --bind /scratch/aplesner   /itet-stor/aplesner/net_scratch/projects_storage/llm-sampling/singularity/vllm.sif   python3 llm-inference.py --input prompts.json --output results.json
```

