# from vlmeval.config import supported_VLM
# model = supported_VLM['GPT4o_20241120']()
# # 前向单张图片
# ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
# print(ret)  # 这张图片上有一个带叶子的红苹果
# # 前向多张图片
# ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
# print(ret)  # 提供的图片中有两个苹果

import re

pattern = r'\b(?:correct|answer|option|final\s*answer|correct\s*answer)\b[^:：]*[:：]\s*([^\.。]*)'
flags = re.IGNORECASE | re.DOTALL

text = """
The Final Answer is: \n A \n
"""

match = re.search(pattern, text, flags=flags)
if match:
    print("提取的答案是:", match.group(1).strip())