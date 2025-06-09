import os
import openai
import backoff 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base
    
#llama setup
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def format_chat_prompt(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    return (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

def llama(user_prompt, system_prompt = "You are a helpful assistant", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    prompt = format_chat_prompt(user_prompt, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = []
    # print(f'LLAMA prompt: \n{prompt}')
    while n > 0:
        with torch.no_grad():
            res = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=0.9
            )
        raw_output_text = tokenizer.decode(res[0], skip_special_tokens=True)
        output_text = raw_output_text
        # print(f'Raw LLAMA output: {output_text}\n')
        if prompt in output_text:
            output_text = output_text.replace(prompt, "")
        if "assistant" in output_text:
            output_text =  output_text[(output_text.index("assistant") + 9):]
        outputs.append(output_text)
        print(f'LLAMA output: {output_text}\n')
        if(output_text == ""):
            print(f'LLAMA output empty, raw output: {raw_output_text}')
        n -= 1
    return outputs

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
