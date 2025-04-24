import re

text = '''
The correct option is:
**A**"
'''
pattern = r'\b(?:correct|answer|option)\b[\s\S]*?([A-D])'


match = re.search(pattern, text)
if match:
    print(match.group(1))  # 输出：A