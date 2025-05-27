#!/bin/bash

start_time=$(date +%s)

# 스크립트가 있는 디렉토리로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# 필요한 디렉토리 확인
if [ ! -d "data/gsm_data" ]; then
    echo "Error: GSM8K 데이터 디렉토리를 찾을 수 없습니다."
    exit 1
fi

# GPU 사용 가능 여부 확인
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU가 감지되지 않습니다."
    exit 1
fi

# 실행
echo "Qwen2.5-7B-Instruct 테스트를 시작합니다..."
echo "데이터셋: GSM8K (3.5% 샘플)"
echo "----------------------------------------"

CUDA_VISIBLE_DEVICES=0,1,2,3 python opro/optimization/optimize_instructions.py \
    --optimizer="qwen2.5-7b-instruct" \
    --scorer="qwen2.5-7b-instruct" \
    --instruction_pos="A_begin" \
    --dataset="gsm8k" \
    --task="train"

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "테스트가 성공적으로 완료되었습니다."
else
    echo "----------------------------------------"
    echo "테스트 실행 중 오류가 발생했습니다."
    exit 1
fi

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "총 실행 시간: ${elapsed}초" 