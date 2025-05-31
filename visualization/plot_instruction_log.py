import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../opro/evaluation')))
from parse_instruction_log import parse_instruction_log


def plot_acc_by_step(log_path: str, save_path: str = None):
    results, _ = parse_instruction_log(log_path)
    steps = sorted(results.keys())
    means = []
    mins = []
    maxs = []
    for step in steps:
        accs = [acc for acc, _ in results[step]]
        means.append(np.mean(accs))
        mins.append(np.min(accs))
        maxs.append(np.max(accs))

    plt.figure(figsize=(6, 4))
    # model_name 추출 (save_path 있을 때만 사용하므로 미리 추출)
    model_name = ""
    if "llama3.1-8b-instruct" in log_path:
        model_name = "llama3.1-8b-instruct"
        plot_model_name = "Llama3.1-8b-Instruct"
    elif "llama3.1-8b" in log_path:
        model_name = "llama3.1-8b"
        plot_model_name = "Llama3.1-8b"
    elif "qwen2.5-7b-instruct" in log_path:
        model_name = "qwen2.5-7b-instruct"
        plot_model_name = "Qwen2.5-7b-Instruct"
    elif "qwen2.5-7b" in log_path:
        model_name = "qwen2.5-7b"
        plot_model_name = "Qwen2.5-7b"
    else:
        raise NotImplementedError(f"Unknown model: {log_path}")

    plt.plot(steps, means, label=f'{plot_model_name}', color='blue', marker='o')
    plt.fill_between(steps, mins, maxs, color='blue', alpha=0.1)
    plt.xlabel('Step', fontsize=16)
    plt.ylabel('Training Accuracy', fontsize=16)
    plt.legend()
    plt.tight_layout()
    
    
    plt.grid(True)
    if save_path:
        dataset_name = ""
        if "GSM8K" in log_path:
            dataset_name = "GSM8K"
        else:
            raise NotImplementedError(f"Unknown dataset: {log_path}")
        
        instruction_type = ""
        if "Q_begin" in log_path:
            instruction_type = "Q_begin"
        elif "A_begin" in log_path:
            instruction_type = "A_begin"
        else:
            raise NotImplementedError(f"Unknown instruction type: {log_path}")
        
        optimization_step = ""
        if "step100" in log_path:
            optimization_step = "step100"
        elif "step200" in log_path:
            optimization_step = "step200"
        else:
            raise NotImplementedError(f"Unknown optimization step: {log_path}")
        
        file_name = f"{dataset_name}_{model_name}_{instruction_type}_{optimization_step}_acc_by_step.pdf"
        os.makedirs(os.path.join(save_path, optimization_step), exist_ok=True)
        plt.savefig(os.path.join(save_path, optimization_step, file_name))
        print(f"그래프가 {os.path.join(save_path, file_name)}에 저장되었습니다.")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="instruction_log.txt 시각화")
    parser.add_argument('--log_path', type=str, required=True, help="instruction_log.txt 파일 경로")
    parser.add_argument('--save_path', type=str, default="../outputs/visualization", help="그래프 저장 경로 (미지정시 화면에 표시)")
    args = parser.parse_args()
    plot_acc_by_step(args.log_path, args.save_path)