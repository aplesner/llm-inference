# Start with the NVIDIA CUDA base image for development on Ubuntu 24.04
# FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04

# # Set environment variable to avoid interactive prompts during apt installations
# ENV DEBIAN_FRONTEND=noninteractive


# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"


FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set environment variable to avoid interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y git

# RUN pip install --no-cache-dir --root-user-action ignore uv 

# RUN uv pip install --no-cache-dir --system --break-system-packages ninja

# RUN uv pip install --no-cache-dir --system --break-system-packages \
#         transformers \
#         datasets \
#         peft \
#         wandb \
#         pyyaml \
#         deepspeed \
#         numpy # \
#         # flash-attn #\ --build-isolation
#         #deepspeed \
#         #vllm==0.8.5


RUN git clone https://github.com/Dao-AILab/flash-attention /flash-attention

WORKDIR /flash-attention/hopper

RUN ls && python setup.py install
