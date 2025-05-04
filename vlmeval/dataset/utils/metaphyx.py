from vlmeval.smp import *
from vlmeval.utils import can_infer
import re
import json
import os
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict
from vlmeval.dataset.utils.step_scorer.utils import *

FAIL_MSG = 'Failed to obtain answer via API.'

#************** Answer Evaluation ****************


def get_ICE():
    example_1 = """
Ground truth answer: 26.7kg \n
Predicted answer: The mass of block \( B \) is: 
\[ 
\boxed{26.7 \, \text\{kg\}}
\] \n
Judegement: 1
"""

    example_2 = """
Ground truth answer: 46.3kN \n
Predicted answer: The tension \( T_B \) in the cable is approximately:
\[
\boxed{46300 \, \text{N}}
\] \n
Judegement: 1
"""

    example_3 = """
Ground truth answer: 12.3m/s \n
Predicted answer: The speed of the box after 2.00 seconds is:
\[
\boxed{12.3 \, \text{m/s}}
\] \n
Judegement: 1
"""

    example_4 = """
Ground truth answer: 36.0kg \n
Predicted answer: The mass of the hanging block \( m_2 \) must be approximately:
\[
\boxed{36.1 \, \text\{kg\}}
\] \n
Judegement: 0
"""

    example_5 = """
Ground truth answer: 0.8m \n
Predicted answer: The distance \( l \) between the forces should be:
\[
\boxed{0.80 \, \text\{m\}}
\] \n
Judegement: 1
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_metaphyx_gpt4_prompt(line):
    task_description = """
Please read the following example. Given predicted answer and ground truth answer, 
compare the these two answers, then ONLY output judegement 1/0 for matched/unmatched at the end of the prompt.\n
"""
    gt_answer = line['answer']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += 'Ground truth answer: {} \n'.format(gt_answer)
    prompt += 'Predicted answer: {} \n'.format(prediction)
    prompt += 'Judegement:'
    return prompt


def MetaPhyX_auxeval(model, line):
    prompt = build_metaphyx_gpt4_prompt(line)
    log = ''
    retry = 5

    gt_answer = line['answer']
    prediction = line['prediction']

    # if "Final Answer:" in prediction:
        # prediction = prediction.split("Final Answer:")[-1]
        # print("hit", gt_answer, "*****", prediction)

    # extract final answer
    pattern = r'\b(?:correct|answer|option|final\s*answer|correct\s*answer)\b[^:：]*[:：]\s*([^\.。]*)'
    flags = re.IGNORECASE | re.DOTALL
    match = re.search(pattern, prediction, flags=flags)
    if match:
        prediction=match.group(1)
    
    # judge via LLM
    if gt_answer.strip().lower() == prediction.strip().lower():
        return dict(log="Matched at string level", res=1)
    
    for i in range(retry):
        res = model.generate(prompt, temperature=i * 0.5)
        if FAIL_MSG in res:
            log += f'Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n'
        else:
            log += 'Compared at semantic level. '
            # print(res)
            if "1" in res or 1 == res:
                log += "Semantic equal via LLM."
                return dict(log=log, res=1)
            elif "0" in res or 0 == res:
                log += "LLM judgement {}".format(res)
                return dict(log=log, res=0)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res=0)


def MetaPhyX_acc(result_file):
    data = load(result_file)
    lt = len(data)
    res = {}
    hit = 0
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        if cate in res.keys():
            res[cate].append(item['res'])
        else:
            res[cate] = [item['res']]
        hit += item['res']
    
    final_res = {}
    final_res["Overall Acc"] = hit/lt
    for k,v in res.items():
        final_res[k] = sum(v)/len(v)
    df = pd.DataFrame(final_res, index=[0])
    return df


def MetaPhyX_process_line(line):
    ret = {}
    if istype(line['answer'], list):
        answers = eval(line['answer'])
    else:
        answers = [line['answer']]


    ret['gt'] = answers
    ret['pred'] = line['prediction'].strip()
    ret['match'] = []
    for x in ret['gt']:
        # TB modify
        # pattern = r'\b(?:correct|answer|option|Correct|Answer|Option)\b[\s\S]*?([A-D])'
        pattern = r'\b(?:correct|answer|option|final\s*answer|correct\s*answer)\b[^:：]*[:：]\s*([^\.。]*)'
        flags = re.IGNORECASE | re.DOTALL
        match = re.search(pattern, ret['pred'], flags=flags)
        # match = re.search(pattern, ret['pred'])
        if match:
            extracted_answer=match.group(1)
            # print(extracted_answer, x)
            # compare string
            if x.strip().lower() == extracted_answer.strip().lower():
                ret['match'].append(1)
                continue
        # 正则匹配不成功，尝试字符比对
        if x+": " in ret['pred']:
            ret['match'].append(1)
        else:
            ret['match'].append(0)
            
    return ret

#************** CoT Evaluation ****************

# extract cot steps
def get_cot_ICE():
    example_1 = """
Model Response: 

To solve this problem, we need to determine the magnitude of the pulls $\vec{A}$ and $\vec{B}$ such that their combined effect produces an outward traction force of 12.8 N on the patient's arm. Here's the step-by-step reasoning:
**Step 1: Understand the Problem**
- The pulls $\vec{A}$ and $\vec{B}$ have equal magnitudes, so let the magnitude of each pull be $F$.
- Both pulls are applied at an angle of $32^\circ$ relative to the horizontal (outward direction).
- The resultant force in the outward (horizontal) direction must be 12.8 N.
**Step 2: Resolve the Forces into Components**
Each pull $\vec{A}$ and $\vec{B}$ can be resolved into horizontal and vertical components:
- For $\vec{A}$:
  - Horizontal component: $A_x = F \cos(32^\circ)$
  - Vertical component: $A_y = F \sin(32^\circ)$
- For $\vec{B}$:
  - Horizontal component: $B_x = F \cos(32^\circ)$
  - Vertical component: $B_y = -F \sin(32^\circ)$ (negative because it is downward)
**Step 3: Sum the Horizontal Components**
The total horizontal force is the sum of the horizontal components of $\vec{A}$ and $\vec{B}$. Since both pulls contribute equally in the horizontal direction:
$$
F_{\text{horizontal}} = A_x + B_x = F \cos(32^\circ) + F \cos(32^\circ) = 2F \cos(32^\circ)
$$
We are given that the total horizontal force must be 12.8 N:
$$
2F \cos(32^\circ) = 12.8
$$
**Step 4: Solve for $F$**
Rearrange the equation to solve for $F$:
$$
F = \frac{12.8}{2 \cos(32^\circ)}
$$
First, calculate $\cos(32^\circ)$:
$$
\cos(32^\circ) \approx 0.848
$$
Now substitute this value into the equation:
$$
F = \frac{12.8}{2 \times 0.848} = \frac{12.8}{1.696} \approx 7.55 \, \text{N}
$$
**Step 5: Verify the Solution**
- The magnitude of each pull is $F \approx 7.55 \, \text{N}$.
- The horizontal component of each pull is:
  $$
  F \cos(32^\circ) \approx 7.55 \times 0.848 \approx 6.4 \, \text{N}
  $$
- The total horizontal force is:
  $$
  2 \times 6.4 = 12.8 \, \text{N}
  $$
This matches the required outward traction force, confirming our solution is correct.
**Final Answer**
$$
\boxed{7.55}
$$"

Extracted Response: 

1. **Given**: Two forces $\vec{A}$ and $\vec{B}$ with equal magnitudes $F$, applied at an angle of $32^\circ$ from the horizontal.
2. **Goal**: Find the magnitude $F$ such that the **horizontal (outward)** resultant force is 12.8 N.
3. **Resolve each force into components**:
   - Horizontal component of each = $F \cos(32^\circ)$
4. **Sum the horizontal components**:
   - Total horizontal force = $2F \cos(32^\circ)$
5. **Set up the equation**:
   $$
   2F \cos(32^\circ) = 12.8
   $$
6. **Solve for $F$**:
   $$
   F = \frac{12.8}{2 \cos(32^\circ)} \approx \frac{12.8}{1.696} \approx 7.55
   $$
**Final Answer:**
$$
\boxed{7.55}\text{N}
$$
"""


    example_2 = """
Model Response: 

To determine how many degrees apart Alkaid and the Sun would appear to an inhabitant of a planet orbiting Merak, we need to calculate the angular separation between Alkaid and the Sun as viewed from Merak. Here's the step-by-step reasoning:
**Step 1: Understand the Problem**
- We are given the distances of various stars from Earth in light-years (ly).
- The distances are:
  - Alkaid: 138 ly
  - Merak: 77 ly
  - Sun: 0 ly (since the Sun is at the origin for this calculation)
- We need to find the angular separation between Alkaid and the Sun as seen from Merak.
**Step 2: Define the Geometry**
- Let the position of the Sun be at the origin $(0, 0, 0)$.
- Let the position of Merak be at $(d_M, 0, 0)$, where $d_M = 77$ ly.
- Let the position of Alkaid be at $(x_A, y_A, z_A)$, where the distance from Earth to Alkaid is 138 ly.
Since the problem does not specify the exact spatial coordinates of Alkaid relative to Merak, we assume that Alkaid lies in the same plane as Merak and the Sun for simplicity. This means Alkaid's position can be described in two dimensions relative to Merak.
**Step 3: Use the Law of Cosines to Find the Angular Separation**
The angular separation $\theta$ between two objects as seen from a third point can be calculated using the law of cosines in spherical trigonometry. However, for small angles (which is typical in astronomy), we can use the simpler formula for angular separation:
$$
\theta \approx \frac{d}{D}
$$
where:
- $d$ is the projected separation between the two objects (Alkaid and the Sun) as seen from Merak.
- $D$ is the distance from Merak to the midpoint of the line segment connecting Alkaid and the Sun.
**Step 3.1: Calculate the Projected Separation $d$**
The projected separation $d$ between Alkaid and the Sun as seen from Merak is the straight-line distance between their positions in the plane of Merak, Alkaid, and the Sun.

Using the distances:
- Distance from Earth to Merak ($d_M$) = 77 ly
- Distance from Earth to Alkaid ($d_A$) = 138 ly

Assume Alkaid and the Sun lie along a straight line with Merak. The projected separation $d$ is simply the difference in distances:
$$
d = d_A - d_M = 138 - 77 = 61 \, \text{ly}
$$
**Step 3.2: Calculate the Angular Separation $\theta$**
The angular separation $\theta$ is given by:
$$
\theta \approx \frac{d}{D}
$$
where $D$ is the distance from Merak to the midpoint of the line segment connecting Alkaid and the Sun. Since Merak is closer to the Sun than to Alkaid, we use the distance from Merak to the Sun ($d_M = 77$ ly) as an approximation for $D$.
Thus:
$$
\theta \approx \frac{d}{d_M} = \frac{61}{77}
$$
Convert this ratio to degrees using the fact that $1 \, \text{radian} \approx 57.3^\circ$:
$$
\theta \approx \frac{61}{77} \times 57.3^\circ
$$
**Step 3.3: Perform the Calculation**
First, calculate the ratio:
$$
\frac{61}{77} \approx 0.7922
$$
Now convert to degrees:
$$
\theta \approx 0.7922 \times 57.3^\circ \approx 45.4^\circ
$$
**Step 4: Final Answer**
The angular separation between Alkaid and the Sun as seen from Merak is approximately:
$$
\boxed{45.4}
$$

Extracted Response: 

1. **Given**: Distances from Earth — Alkaid = 138 ly, Merak = 77 ly, Sun = 0 ly.
2. **Goal**: Find angular separation between Alkaid and the Sun **as seen from Merak**.
3. **Find separation**: $ d = 138 - 77 = 61 $ ly (projected distance between Alkaid and Sun).
4. **Use angular size formula**:  
   $$
   \theta \approx \frac{d}{D} = \frac{61}{77} \approx 0.7922 \text{ radians}
   $$
5. **Convert to degrees**:  
   $$
   \theta \approx 0.7922 \times 57.3^\circ \approx 45.4^\circ
   $$
Final Answer:
$$
\boxed{45.4^\circ}
$$

"""
    return [example_1, example_2]


def get_cot_score_ICE():
    example_1 = """
[Question]: A ball moves in a straight line (the x-axis). The graph  shows this ball velocity as a function of time. What are the ball's average speed  during the first 3.0 s? Please answer the question with step by step reasoning.
[Standard Answer]: 2.33 m/s
[Model Response]: 
1. Goal: Find **average speed** in 3.0 s → use position graph.
2. Convert 3.0 s = 0.05 min.
3. From graph: ball moves from 0 m to ~10 m.
4. Total distance = 10 m.
5. Avg speed = $ \frac{10\, \text{m}}{3.0\, \text{s}} \approx 3.33\, \text{m/s} $
Final Answer:
$$
\boxed{3.33}
$$
Score: 4
    """

    example_2 = """
[Question]: A ball is thrown vertically upward at the same instant that a second ball is dropped from rest directly above it. The two balls are  12.0m apart when they start their motion. Find the maximum speed at which the first ball can be thrown such that it doesn't collide with the second ball before it returns to its starting height. Treat the balls as being very small (i.e. ignore their diameters).
[Standard Answer]: 7.67 m/s
[Model Response]: 
1. Ball A is thrown upward, returns in time $ t = \frac{2v_0}{g} $
2. Ball B falls distance: $ d_B = \frac{2v_0^2}{g} $
3. To avoid collision: $ d_B \leq 12.0\, \text{m} $
4. Solve: $ \frac{2v_0^2}{g} \leq 12.0 $
5. Max speed: $ v_0 \leq \sqrt{58.8} \approx 7.67\, \text{m/s} $
Final Answer:
$$
\boxed{7.67}\, \text{m/s}
$$
Score: 10
"""

    return [example_1, example_2]

def build_metaphyx_extract_prompt(line):
    task_description = """
I am providing you a response from a model to a physicx problem, termed 'Model Response'. You should extract the key thinking steps and final answer from the response as 'Extracted Response'. Directly output the extracted response with no explanation.\n\n
""" # noqa
    prediction = str(line['prediction'])
    demo_prompt = task_description
    examples = get_cot_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model Response: \n'{prediction}'\nExtracted Response: \n"
    full_prompt = f'{demo_prompt}\n{test_prompt}'

    return full_prompt

def build_metaphyx_score_prompt(line):
    task_description = """
Below are two answers to a physic question. 
Question is [Question], [Standard Answer] is the ground truth answer to the question, and [Model Response] is the thinking steps and final answer from a model's output to this question.  
Base on the [Question] and [Standard Answer], score the [Model Response] in 0-10 considering the Reasoning Consistency, Answer correctness, and Simplicity.
10 for PERFECT Response and 0 for NONSENSE Response, output the score ONLY.\n\n
""" # noqa
    question_for_eval = line['question']
    extract = line['extract']
    answer = line['answer']
    demo_prompt = task_description
    examples = get_cot_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    [Question]: {question_for_eval}
    [Standard Answer]: {answer}
    [Model Response]: \n {extract}
    Score:
    """
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt

def MetaPhyX_cot_extract(model, line):
    prompt = build_metaphyx_extract_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_extract=log, extract=res)
    log += 'All 5 retries failed.\n'
    return dict(log_extract=log, extract='')

def post_check_score(line, prefetch=False):
    ans = str(line['answer']).strip()
    response = str(line['extract']).strip()

    if response == ans:
        return response if prefetch else True
    else:
        return False

def MetaPhyX_cot_score(model, line):
    # score: 0-10
    prompt = build_metaphyx_score_prompt(line)
    log = ''
    retry = 5
    if post_check_score(line, prefetch=True):
        res = post_check_score(line, prefetch=True)
        return dict(log_score='Prefetch succeed', score=10)
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, res is {res}, failed to parse.\n'
        else:
            match = re.search(r'score.*?(\d+\.?\d*)', res, re.I)
            if match:
                try:
                    score = float(match.group(1))
                except:
                    log += f'Try {i}: output is {prediction}, res is {res}, failed to parse.\n'
                    continue
                if score >= 0 and score <= 10:
                    log += 'Succeed'
                    return dict(log_score=log, score=score)
    log += 'All 5 retries failed.\n'
    return dict(log_score=log, score=0)

def MetaPhyX_cot_acc(result_file):
    data = load(result_file)
    lt = len(data)
    res = {}
    score_all = 0
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        if cate in res.keys():
            res[cate].append(item['score'])
        else:
            res[cate] = [item['score']]
        score_all += item['score']
    
    final_res = {}
    final_res["Overall Score"] = score_all/lt
    for k,v in res.items():
        final_res[k] = sum(v)/len(v)
    df = pd.DataFrame(final_res, index=[0])
    return df