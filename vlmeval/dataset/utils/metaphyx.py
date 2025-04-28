from ...smp import *
from ...utils import can_infer
import re


FAIL_MSG = 'Failed to obtain answer via API.'


def get_gpt4_ICE():
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
    examples = get_gpt4_ICE()
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

    if "Final Answer:" in prediction:
        prediction = prediction.split("Final Answer:")[-1]
        # print("hit", gt_answer, "*****", prediction)

    if gt_answer == prediction:
        return dict(log="Matched at string level", res=1)
    for i in range(retry):
        res = model.generate(prompt, temperature=i * 0.5)
        if FAIL_MSG in res:
            log += f'Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n'
        else:
            log += 'Compared at semantic level. '
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
        pattern = r'\b(?:correct|answer|option|Correct|Answer|Option)\b[\s\S]*?([A-D])'
        match = re.search(pattern, ret['pred'])
        if match:
            extracted_answer=match.group(1)
            if x.strip().lower() == extracted_answer.strip().lower():
                ret['match'].append(1)
                continue
        # 正则匹配不成功，尝试字符比对
        if x+": " in ret['pred']:
            ret['match'].append(1)
        else:
            ret['match'].append(0)
            
    return ret
