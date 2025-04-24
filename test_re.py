import re

text = "The instantaneous velocity at point G is the slope of the tangent to the curve at that point. From the graph, the slope at G is **-1**. **Answer: C**"
pattern = r'\b(?:correct|answer|option)\s*[：:\s]*([A-D])'


match = re.search(pattern, text)
if match:
    print(match.group(1))  # 输出：A