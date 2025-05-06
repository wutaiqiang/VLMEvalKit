import re

text = '''
To find the magnitude of the airplane's displacement, we use the law of cosines. The displacement forms a triangle with sides \( d_1 \), \( d_2 \), and the angle between them \( \Delta\theta = 123^\circ \). The formula for the magnitude of the displacement \( s \) is: \[ s = \sqrt{d_1^2 + d_2^2 - 2d_1d_2\cos(\Delta\theta)} \] Substitute the given values: \[ d_1 = 360\ \text{m}, \quad d_2 = 790\ \text{m}, \quad \Delta\theta = 123^\circ \] Convert \( \Delta\theta \) to radians: \[ \Delta\theta = 123^\circ \times \frac{\pi}{180} = 2.1468\ \text{radians} \] Now calculate \( s \): \[ s = \sqrt{360^2 + 790^2 - 2(360)(790)\cos(2.1468)} \] First, calculate each term: \[ 360^2 = 129600, \quad 790^2 = 624100, \quad 2(360)(790) = 568800 \] \[ \cos(2.1468) \approx -0.5446 \] \[ s = \sqrt{129600 + 624100 - 568800(-0.5446)} \] \[ s = \sqrt{129600 + 624100 + 309711.68} \] \[ s = \sqrt{1063411.68} \approx 1031\ \text{m} \] The correct option is: **C: 
'''
pattern = r'\b(?:correct|answer|option)\b[\s\S]*?([A-D])'


match = re.search(pattern, text)
if match:
    print(match.group(1))  # 输出：A