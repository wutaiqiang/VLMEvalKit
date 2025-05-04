# MetaPhyX 测试指南

可以先看看框架的官方说明: 

加数据集说明
https://github.com/wutaiqiang/VLMEvalKit/blob/main/docs/zh-CN/Development.md

使用指南
https://github.com/wutaiqiang/VLMEvalKit/blob/main/docs/zh-CN/README_zh-CN.md


# MetaPhyX 流程

## 准备数据
从 https://github.com/NastyMarcus/MetaPhyX/tree/main/data_tsv 抓取 TSV 文件

移到本目录下的 LMUData, 对应的 MD5 值在 data2tsv.py 的头部区域的注释里

将 MD5值 放到 vlmeval/dataset/image_vqa.py 的 MetaPhyX 类函数里面的 DATASET_MD5

注意，不同的 setting 是不同的文件，放到 LMUData 以后，调用不同的文件名来拉起不同 setting

DATASET_URL 里面的网址可以乱写，后期传给了官方服务器以后再 update真正的

## 拉起任务

参考run_example.sh

--model 是测评的模型, 可以是API，官方支持的模型在https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb 可见
--data 是测试用的文件，不写.tsv 后缀
--judge 指的是用调用判定模型来判定结果准确与否，目前使用SiliconFlow_API_KEY 对应的 Deepseek-V3 来判定，只需要设置SiliconFlow_API_KEY就好了，可以直接用
--judge-args 设置的是额外的参数，比如 valid_type: STR/LLM, step_score: evaluate 过程步

在A100 机器上，需要使用 GPU时，需要设置好CUDA_VISIBLE_DEVICES，然后 sbatch run.sh 日志在 logs 文件夹，结果在 outputs 文件夹

在本地机器，直接 bash run.sh 即可


## TODO

整体的函数实现，在vlmeval/dataset/image_vqa.py的MetaPhyX类

其中，输入处理在build_prompt，输出处理在evaluate

有两种评估方式，字符匹配/模型判定，目前是根据数据集名字来区分的，MC ，也就是选择题，用的是字符匹配，其他用模型，

- [ ] 两种测评方式的改进，借鉴 MMLU 改进字符匹配算法，借鉴 MathVerse 改进模型判定（引入思维链对比), by Wendong
- [ ] 加入 OpenAI 最新的 api 的支持，对应--model 参数
- [x] 使用参数来控制调用不同测评方式，借助**judge_kwargs, by Taiqiang

注：openai 的测评需要梯子，建议到本地运行，反正也不需要拉起 GPU，本地跑就好了
