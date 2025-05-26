from typing import Dict, Any, Union, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelConfig:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map={"": "cpu"}
        )
        return self.model, self.tokenizer

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # 단일 프롬프트를 리스트로 변환
        if isinstance(prompts, str):
            prompts = [prompts]
        
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        attention_mask = inputs['attention_mask']
        input_lengths = attention_mask.sum(dim=1).tolist()  # 각 입력의 실제 길이
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **kwargs
            )
            outputs.to("cpu")

        generated_only = []
        for output, prompt_len in zip(outputs, input_lengths):
            new_tokens = output[prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_only.append(text)

        return generated_only

# 사용 가능한 모델 설정
AVAILABLE_MODELS = {
    "llama2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "llama2-13B": "meta-llama/Llama-2-13b-chat-hf",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "gpt2": "openai-community/gpt2"
}

def get_model(model_name: str) -> ModelConfig:
    """지정된 모델의 설정을 반환합니다."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    return ModelConfig(AVAILABLE_MODELS[model_name]) 