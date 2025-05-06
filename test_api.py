# from vlmeval.config import supported_VLM
# model = supported_VLM['GPT4o_20241120']()
# # 前向单张图片
# ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
# print(ret)  # 这张图片上有一个带叶子的红苹果
# # 前向多张图片
# ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
# print(ret)  # 提供的图片中有两个苹果

import re

pattern = r'\b(?:correct|answer|option|final\s*answer|correct\s*answer)\b[^:：]*[:：]\s*(.*?)(?=\n\n|\Z)'
flags = re.IGNORECASE | re.DOTALL

text = """
To determine the angle \( \theta' \) (the angle of refraction in water), we will use **Snell's Law**, which states: \[ n_1 \sin \theta_1 = n_2 \sin \theta_2 \] Where: - \( n_1 \) and \( n_2 \) are the indices of refraction of the two media, - \( \theta_1 \) and \( \theta_2 \) are the angles of incidence and refraction, respectively. ### Step 1: Identify the given values - The index of refraction of linseed oil (\( n_1 \)) is \( 1.48 \), - The index of refraction of water (\( n_2 \)) is \( 1.33 \), - The angle of incidence in linseed oil (\( \phi = 20.0^\circ \)). We are tasked with finding the angle of refraction in water (\( \theta' \)). --- ### Step 2: Apply Snell's Law at the linseed oil-water interface At the interface between linseed oil and water, Snell's Law is: \[ n_1 \sin \phi = n_2 \sin \theta' \] Substitute the known values: \[ 1.48 \sin(20.0^\circ) = 1.33 \sin \theta' \] --- ### Step 3: Solve for \( \sin \theta' \) First, calculate \( \sin(20.0^\circ) \): \[ \sin(20.0^\circ) \approx 0.3420 \] Substitute this into the equation: \[ 1.48 \times 0.3420 = 1.33 \sin \theta' \] \[ 0.5062 = 1.33 \sin \theta' \] Solve for \( \sin \theta' \): \[ \sin \theta' = \frac{0.5062}{1.33} \approx 0.3807 \] --- ### Step 4: Find \( \theta' \) using the inverse sine function \[ \theta' = \arcsin(0.3807) \] Using a calculator: \[ \theta' \approx 22.4^\circ \] --- ### Final Answer: The angle of refraction in water is: \[ \\boxed{\\theta' \\approx 22.4^\circ} \]
"""

match = re.search(pattern, text, flags=flags)
if match:
    print("提取的答案是:", match.group(1).strip())