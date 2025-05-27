from typing import Dict, Any, Union, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import os


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
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
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

class VLLMConfig:
    def __init__(self, model_name: str, dtype: str = "float16", gpus: str = None):
        self.model_name = model_name
        self.dtype = dtype
        self.llm = None
        self.gpus = gpus

    def load_model(self):
        # 사용할 GPU 지정
        if self.gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
            num_gpus = len(self.gpus.split(","))
        else:
            num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("사용 가능한 GPU가 없습니다.")
        self.llm = LLM(
            model=self.model_name,
            dtype=self.dtype,
            tensor_parallel_size=num_gpus
        )
        return self.llm

    def generate(self, prompts, **kwargs):
        temperature = kwargs.get("temperature", 0.8)
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

# 사용 가능한 모델 설정
AVAILABLE_LOCAL_MODELS = {
    "llama2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "gpt2": "openai-community/gpt2",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B"
}

def get_model(model_name: str, is_vllm: bool = False, gpus: str = None) -> Any:
    if model_name not in AVAILABLE_LOCAL_MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    if is_vllm:
        return VLLMConfig(AVAILABLE_LOCAL_MODELS[model_name], gpus=gpus)
    else:
        return ModelConfig(AVAILABLE_LOCAL_MODELS[model_name])