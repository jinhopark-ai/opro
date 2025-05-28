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
echo "최적의 prompt 테스트를 시작합니다..."
echo "데이터셋: GSM8K Test"
echo "----------------------------------------"

CUDA_VISIBLE_DEVICES=0 python opro/evaluation/evaluate_instructions.py \
    --scorer="llama3.1-8b-instruct" \
    --dataset="gsm8k" \
    --task="test" \
    --instruction_pos="Q_begin" \
    --evaluate_training_fold=false \
    --evaluate_test_fold=true \
    --gpus="0" &

CUDA_VISIBLE_DEVICES=1 python opro/evaluation/evaluate_instructions.py \
    --scorer="llama3.1-8b" \
    --dataset="gsm8k" \
    --task="test" \
    --instruction_pos="A_begin" \
    --evaluate_training_fold=false \
    --evaluate_test_fold=true \
    --gpus="1" &

CUDA_VISIBLE_DEVICES=2 python opro/evaluation/evaluate_instructions.py \
    --scorer="qwen2.5-7b-instruct" \
    --dataset="gsm8k" \
    --task="test" \
    --instruction_pos="Q_begin" \
    --evaluate_training_fold=false \
    --evaluate_test_fold=true \
    --gpus="2" &

CUDA_VISIBLE_DEVICES=3 python opro/evaluation/evaluate_instructions.py \
    --scorer="qwen2.5-7b" \
    --dataset="gsm8k" \
    --task="test" \
    --instruction_pos="A_begin" \
    --evaluate_training_fold=false \
    --evaluate_test_fold=true \
    --gpus="3" &

wait

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

# 시:분:초 형식으로 변환
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))

echo "총 실행 시간: ${hours}시간 ${minutes}분 ${seconds}초"