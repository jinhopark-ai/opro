# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The utility functions for prompting GPT and Google Cloud models."""

import time
import google.generativeai as palm
import openai
from typing import Any, Dict, List, Optional, Tuple, Union
from opro.models import get_model
import torch
from accelerate import infer_auto_device_map, dispatch_model
import math
from tqdm import tqdm


def call_openai_server_single_prompt(
    prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with an input string."""
  try:
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_decode_steps,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content

  except openai.error.Timeout as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.RateLimitError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"API error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIConnectionError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"API connection error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.ServiceUnavailableError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Service unavailable. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except OSError as e:
    retry_time = 5  # Adjust the retry time as needed
    print(
        f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
    )
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )


def call_openai_server_func(
    inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with a list of input strings."""
  if isinstance(inputs, str):
    inputs = [inputs]
  outputs = []
  for input_str in inputs:
    output = call_openai_server_single_prompt(
        input_str,
        model=model,
        max_decode_steps=max_decode_steps,
        temperature=temperature,
    )
    outputs.append(output)
  return outputs


def call_palm_server_from_cloud(
    input_text, model="text-bison-001", max_decode_steps=20, temperature=0.8
):
  """Calling the text-bison model from Cloud API."""
  assert isinstance(input_text, str)
  assert model == "text-bison-001"
  all_model_names = [
      m
      for m in palm.list_models()
      if "generateText" in m.supported_generation_methods
  ]
  model_name = all_model_names[0].name
  try:
    completion = palm.generate_text(
        model=model_name,
        prompt=input_text,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
    )
    output_text = completion.result
    return [output_text]
  except:  # pylint: disable=bare-except
    retry_time = 10  # Adjust the retry time as needed
    print(f"Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_palm_server_from_cloud(
        input_text, max_decode_steps=max_decode_steps, temperature=temperature
    )


def call_openai_server_batch(
    prompts, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
    """OpenAI 배치 엔드포인트를 사용하여 여러 프롬프트를 한 번에 처리합니다."""
    if isinstance(prompts, str):
        prompts = [prompts]
    messages = [
        {"role": "user", "content": prompt}
        for prompt in prompts
    ]
    try:
        # openai.ChatCompletion.create의 경우, messages에 여러 개의 대화를 넣는 것이 아니라
        # 각각의 프롬프트를 별도의 요청으로 만들어야 합니다.
        # 공식 배치 API가 지원된다면 아래와 같이 사용합니다.
        response = openai.ChatCompletion.create(
            model=model,
            messages=[messages],  # 여러 개의 messages 리스트를 감싸서 전달
            temperature=temperature,
            max_tokens=max_decode_steps,
            # batch_size 등 추가 파라미터가 필요할 수 있음
        )
        # 응답에서 각 프롬프트에 대한 결과 추출
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"Error during batch inference: {e}")
        raise
      

def call_local_model_server(
    prompts: Union[str, List[str]],
    model_type: str,
    temperature: float = 0.0,
    max_decode_steps: int = 1024,
    batch_size: int = 1,
    is_vllm: bool = False,
    **kwargs
) -> list:
    print(f"\n[DEBUG] In call_local_model_server function")
    model_config = get_model(model_type, is_vllm=is_vllm, gpus=kwargs.pop("gpus", None))
    print(f"[DEBUG] model_config: {model_config}")
    model_config.load_model()
        

    print(f"[DEBUG] Generating response with temperature={temperature}, max_length={max_decode_steps}")
    
    # 단일 프롬프트를 리스트로 변환
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Hugging Face 모델의 generation_config 설정
    if is_vllm:
        generation_config = {
            "max_new_tokens": max_decode_steps,
            "temperature": temperature
        }
    else:
        generation_config = {
            "max_new_tokens": max_decode_steps,
            "do_sample": temperature > 0,  # temperature가 0보다 크면 sampling 활성화
            "temperature": temperature if temperature > 0 else None,  # temperature가 0이면 None으로 설정
            "pad_token_id": model_config.tokenizer.pad_token_id,
            "eos_token_id": model_config.tokenizer.eos_token_id,
        }
    try:
        if 'batch_size' in kwargs:
            batch_size = kwargs.pop('batch_size')
            all_responses = []
            
            # 배치 단위로 프롬프트 처리
            for i in tqdm(range(0, len(prompts), batch_size)):
                batch_prompts = prompts[i:i + batch_size]
                batch_responses = model_config.generate(
                    batch_prompts,
                    **generation_config,
                    **kwargs
                )
                all_responses.extend(batch_responses)
        else:
            all_responses = model_config.generate(
                prompts,
                **generation_config,
                **kwargs
            )
            # GPU에서 생성된 결과를 CPU로 이동
            if hasattr(all_responses, 'cpu'):
                all_responses = all_responses.cpu()
        
    except Exception as e:
        print(f"[ERROR] 모델 생성 중 오류 발생: {e}")
        raise
    return all_responses
