import re
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse


def parse_instruction_log(filepath: str):
    """
    instruction_log.txt 파일을 'Step'으로 시작하는 줄을 기준으로 분리하여
    각 step별로 (acc, instruction) 튜플의 리스트를 반환하고,
    전체에서 가장 높은 acc와 해당 instruction도 추가로 반환합니다.

    Args:
        filepath (str): instruction_log.txt 파일 경로
    Returns:
        Tuple[Dict[int, List[Tuple[float, str]]], Tuple[float, str]]: 
            (step별 (acc, instruction) 리스트 딕셔너리, (최대 acc, 해당 instruction))
    """
    step_pattern = re.compile(r"Step (\d+), training acc: ([0-9.]+), instruction:([\s\S]*?)(?=\nStep |\Z)")
    results = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        file_context = f.read()
    matches = step_pattern.findall(file_context)
    for match in matches:
        step = int(match[0])
        acc = float(match[1])
        instruction = match[2]
        results[step].append((acc, instruction))
    # 전체에서 가장 높은 acc와 해당 instruction 찾기
    max_acc = -float('inf')
    max_instr = None
    for acc_instr_list in results.values():
        for acc, instr in acc_instr_list:
            if instr.strip() == "" or instr.strip() == "Let's think step by step.":
                continue
            if acc > max_acc:
                max_acc = acc
                max_instr = instr
    return dict(results), (max_acc, max_instr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="instruction_log.txt 파싱")
    parser.add_argument('--log_path', type=str, default="", help="instruction_log.txt 파일 경로")
    args = parser.parse_args()

    result_dict, (max_acc, max_instr) = parse_instruction_log(args.log_path)
    print(f"전체 최대 acc: {max_acc}, instruction: {max_instr.strip()}")
    