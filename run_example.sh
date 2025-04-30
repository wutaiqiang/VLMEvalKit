export LMUData="/data/home/takiwu/VLMEvalKit/LMUData"

export SiliconFlow_API_KEY=sk-xxx

cd /data/home/takiwu/VLMEvalKit

# python -u run.py --data MetaPhyX_MC --model GPT4o_20241120 --reuse

python -u run.py --data MetaPhyX --model GPT4o_20241120 --judge deepseek  --reuse