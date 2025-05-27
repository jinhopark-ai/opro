from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    for model_name in ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        prompt = "What is the capital of France?"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    

if __name__=="__main__":
    main()