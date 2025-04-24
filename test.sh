#!/bin/bash

#SBATCH  -J llava_ft
#SBATCH  -N 1
#SBATCH  --gres=gpu:1
#SBATCH  --output=logs/benchmark_eva_log_%j_all.out
#SBATCH  --error=logs/benchmark_eva_log_%j_all.out
#SBATCH  --time=24:00:00
#SBATCH  --partition=LocalQ
#SBATCH  --ntasks=1

source /home/tqwu/anaconda3/bin/activate xtuner-env

nvidia-smi

cd /home/tqwu/resource_dir/VLMEvalKit

export CUDA_VISIBLE_DEVICES=0

# pip install einops timm accelerate-1.6.0

# MMBench_DEV_CN MMBench_TEST_CN AI2D_TEST ScienceQA_TEST
# MathVerse_MINI

#! 这个 api 免费送的，后期得换成可报销的，收费版

python -u run.py \
    --data MetaPhyX_MC \
    --model GPT4o_HIGH

# python -u run.py \
#     --data  MetaPhyX \
#     --model GPT4o_HIGH --judge gpt-4o 

# deepseek 调用的是  SiliconFlow的接口，对应的key的环境变量是  SiliconFlow_API_KEY