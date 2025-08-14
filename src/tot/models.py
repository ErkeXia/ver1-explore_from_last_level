# import os
# import openai
# import backoff 
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from text_generation import Client
# import flash_attn


# #TGI endpoint
# client = Client("http://localhost:8080", timeout=60)

# completion_tokens = prompt_tokens = 0
# llama_tokens = prompt_llama_tokens = 0
# cached_system_prompts = {}

# model_name = None
# model = None
# tokenizer = None
# TGI = False
# has_load = False

# api_key = os.getenv("OPENAI_API_KEY", "")
# if api_key != "":
#     openai.api_key = api_key
# else:
#     print("Warning: OPENAI_API_KEY is not set")
    
# api_base = os.getenv("OPENAI_API_BASE", "")
# if api_base != "":
#     print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
#     openai.api_base = api_base
    
    
# def model_setup(model_name_arg, TGI_arg = False):
#     global tokenizer, model, model_name, TGI, has_load
    
#     TGI = TGI_arg
#     model_name = model_name_arg
#     if TGI or has_load:
#         return
    
#     torch.cuda.empty_cache()
#     if model_name == 'llama':
#         model_id = "meta-llama/Llama-3.1-8B-Instruct"
#     elif model_name == 'mistral':
#         model_id = "mistralai/Mistral-7B-Instruct-v0.2"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         use_cache=False,
#         attn_implementation="flash_attention_2"
#         # attn_implementation="flash_attention_2"
#     )
#     model = torch.compile(model)
    
#     # tokenizer.pad_token = tokenizer.eos_token
#     # model.config.pad_token_id = tokenizer.eos_token_id
#     # tokenizer.padding_side = "left"
    
#     has_load = True

# #llama setup

# def format_chat_prompt_TGI(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
#     if model_name == 'llama':
#         return (
#             "<|start_header_id|>system<|end_header_id|>\n"
#             f"{system_prompt}<|eot_id|>\n"
#             "<|start_header_id|>user<|end_header_id|>\n"
#             f"{user_prompt}<|eot_id|>\n"
#             "<|start_header_id|>assistant<|end_header_id|>\n"
#         )
#     elif model_name == 'mistral':
#         return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt} [/INST]"
    
# def format_chat_prompt_B(user_prompts, system_prompt: str = "You are a helpful assistant.") -> tuple[str, str]:
#     if model_name == 'llama':
#         system_part = (
#             "<|start_header_id|>system<|end_header_id|>\n"
#             f"{system_prompt}<|eot_id|>\n"
#         )
#         user_parts = [(
#             "<|start_header_id|>user<|end_header_id|>\n"
#             f"{user_prompt}<|eot_id|>\n"
#             "<|start_header_id|>assistant<|end_header_id|>\n"
#         ) for user_prompt in user_prompts]
#     elif model_name == 'mistral':
#         system_part = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
#         user_parts = [f"{user_prompt} [/INST]" for user_prompt in user_prompts]
#     return system_part, user_parts

# def format_chat_prompt(user_prompt, system_prompt: str = "You are a helpful assistant.") -> tuple[str, str]:
#     if model_name == 'llama':
#         system_part = (
#             "<|start_header_id|>system<|end_header_id|>\n"
#             f"{system_prompt}<|eot_id|>\n"
#         )
#         user_part = (
#             "<|start_header_id|>user<|end_header_id|>\n"
#             f"{user_prompt}<|eot_id|>\n"
#             "<|start_header_id|>assistant<|end_header_id|>\n"
#         )
#     elif model_name == 'mistral':
#         system_part = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
#         user_part = f"{user_prompt} [/INST]"
#     return system_part, user_part


# def llama_B(user_prompts, system_prompt = "You are a helpful assistant", temperature=0.7, max_tokens=1000, n=1, stop=None, query_task = None) -> list:
#     global prompt_llama_tokens, llama_tokens
#     # print(model.config._attn_implementation)

#     outputs = []
    
#     if TGI:
#         for user_prompt in user_prompts:
#             TGI_prompt = format_chat_prompt_TGI(user_prompt, system_prompt)
#             for _ in range(n):
#                 response = client.generate(
#                     TGI_prompt,
#                     max_new_tokens=max_tokens,
#                     temperature=temperature,
#                     top_p=0.9,
#                     do_sample=True,
#                     return_full_text=False
#                 )
#                 output_text = response.generated_text

#                 if output_text.strip() == "":
#                     print(f"TGI output empty for prompt: {prompt}")
#                 outputs.append(output_text)
#         return outputs
    
#     prompts = [format_chat_prompt_TGI(system_prompt, u) for u in user_prompts]
#     inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
#     input_seq_len = inputs.input_ids.shape[1]
#     prompt_lens = inputs.attention_mask.sum(dim=1).tolist()
    
#     with torch.no_grad():
#         res = model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             temperature=temperature,
#             top_p=0.9,
#             num_return_sequences=n,
#         )
        
#     B = inputs.input_ids.size(0)

#     for i in range(B):
#         per_input = []
#         prompt_llama_tokens += int(prompt_lens[i])
#         for j in range(n):
#             row = i * n + j
#             gen = res[row]
            
#             gen_only = gen[input_seq_len:]
#             text = tokenizer.decode(gen_only, skip_special_tokens=True)
#             per_input.append(text)
            
#             generated_token_count = int(gen.shape[-1] - input_seq_len)
#             llama_tokens += generated_token_count
            
#         outputs.append(per_input)

#     return outputs

# def llama(user_prompt, system_prompt = "You are a helpful assistant", temperature=0.7, max_tokens=1000, n=1, stop=None, query_task = None) -> list:
#     global prompt_llama_tokens, llama_tokens
#     # print(model.config._attn_implementation)

#     outputs = []
    
#     if TGI:
#         TGI_prompt = format_chat_prompt_TGI(user_prompt, system_prompt)
#         for _ in range(n):
#             response = client.generate(
#                 TGI_prompt,
#                 max_new_tokens=max_tokens,
#                 temperature=temperature,
#                 top_p=0.9,
#                 do_sample=True,
#                 return_full_text=False
#             )
#             output_text = response.generated_text

#             if output_text.strip() == "":
#                 print(f"TGI output empty for prompt: {prompt}")
#             outputs.append(output_text)
#         return outputs
    
#     system_part, user_part = format_chat_prompt(user_prompt, system_prompt)
    
#     if query_task != None and query_task in cached_system_prompts:
#         system_tokens_cache = cached_system_prompts[query_task]
#     else:
#         system_tokens_cache = tokenizer(system_part, return_tensors="pt", add_special_tokens=False).input_ids
#         cached_system_prompts[query_task] = system_tokens_cache
#     user_tokens = tokenizer(user_part, return_tensors="pt", add_special_tokens=False).input_ids
#     full_tokens = torch.cat([system_tokens_cache, user_tokens], dim=-1).to(model.device)
#     attention_mask = torch.ones_like(full_tokens)
#     inputs = {
#         "input_ids": full_tokens.to(model.device),
#         "attention_mask": attention_mask.to(model.device)
#     }

#     # print(f'LLAMA prompt: \n{prompt}')
#     prompt_len = inputs["input_ids"].shape[1]
#     prompt_llama_tokens += prompt_len
#     # print(f'len {prompt_len} \n prompt tokens {prompt_token_count}')
#     with torch.no_grad():
#         res = model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             temperature=temperature,
#             top_p=0.9,
#             num_return_sequences=n,
#         )
#         # raw_output_text = tokenizer.decode(res[0], skip_special_tokens=True)
#     for output in res:
#         generated_tokens = output[prompt_len:]
#         raw_output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#         output_text = raw_output_text
#         outputs.append(output_text)
#         generated_token_count = output.shape[-1] - prompt_len
#         llama_tokens += generated_token_count
#             # print(f'LLAMA output: {output_text}\n')
#         if(output_text == ""):
#             print(f'LLAMA output empty, raw output: {raw_output_text}')
#     return outputs


# @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
# def completions_with_backoff(**kwargs):
#     return openai.ChatCompletion.create(**kwargs)

# def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
#     messages = [{"role": "user", "content": prompt}]
#     return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
# def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
#     global completion_tokens, prompt_tokens
#     # print(f"Messages: {messages}")
#     outputs = []
#     while n > 0:
#         cnt = min(n, 20)
#         n -= cnt
#         res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
#         outputs.extend([choice.message.content for choice in res.choices])
#         # log completion tokens
#         completion_tokens += res.usage.completion_tokens
#         prompt_tokens += res.usage.prompt_tokens
#     return outputs

# def reset():
#     global completion_tokens, prompt_tokens, llama_tokens, prompt_llama_tokens
#     completion_tokens = prompt_tokens = 0
#     llama_tokens = prompt_llama_tokens = 0
    
# def gpt_usage(backend="gpt-4"):
#     global completion_tokens, prompt_tokens
#     if backend == "gpt-4":
#         cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
#     elif backend == "gpt-3.5-turbo":
#         cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
#     elif backend == "gpt-4o":
#         cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
#     return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

# def llama_usage():
#     global prompt_llama_tokens, llama_tokens
#     return {"llama_completion_tokens": llama_tokens, "llama_prompt_tokens": prompt_llama_tokens}











import os
import openai
import backoff 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from text_generation import Client
import flash_attn


#TGI endpoint
client = Client("http://localhost:8080", timeout=60)

completion_tokens = prompt_tokens = 0
llama_tokens = prompt_llama_tokens = 0
cached_system_prompts = {}

model_name = None
model = None
tokenizer = None
TGI = False
has_load = False

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base
    
    
def model_setup(model_name_arg, TGI_arg = True):
    global tokenizer, model, model_name, TGI, has_load
    
    TGI = TGI_arg
    model_name = model_name_arg
    if TGI or has_load:
        return
    
    torch.cuda.empty_cache()
    if model_name == 'llama':
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
    elif model_name == 'mistral':
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False,
        attn_implementation="flash_attention_2"
        # attn_implementation="flash_attention_2"
    )
    model = torch.compile(model)
    has_load = True

#llama setup

def format_chat_prompt_TGI(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    if model_name == 'llama':
        return (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    elif model_name == 'mistral':
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt} [/INST]"
    
def format_chat_prompt(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> tuple[str, str]:
    if model_name == 'llama':
        system_part = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|>\n"
        )
        user_part = (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    elif model_name == 'mistral':
        system_part = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
        user_part = f"{user_prompt} [/INST]"
    return system_part, user_part


def llama(user_prompt, system_prompt = "You are a helpful assistant", temperature=0.7, max_tokens=1000, n=1, stop=None, query_task = None) -> list:
    global prompt_llama_tokens, llama_tokens
    # print(model.config._attn_implementation)

    outputs = []
    
    if TGI:
        TGI_prompt = format_chat_prompt_TGI(user_prompt, system_prompt)
        for _ in range(n):
            response = client.generate(
                TGI_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )
            output_text = response.generated_text

            # prompt_token_count = len(tokenizer.encode(prompt))
            # completion_token_count = len(tokenizer.encode(output_text))
            # prompt_llama_tokens += prompt_token_count
            # llama_tokens += completion_token_count

            if output_text.strip() == "":
                print(f"TGI output empty for prompt: {prompt}")
            outputs.append(output_text)
        return outputs
    
    system_part, user_part = format_chat_prompt(user_prompt, system_prompt)
    # print(f"system: {system_part} user: {user_part}")
    
    if query_task != None and query_task in cached_system_prompts:
        system_tokens_cache = cached_system_prompts[query_task]
    else:
        system_tokens_cache = tokenizer(system_part, return_tensors="pt", add_special_tokens=False).input_ids
        cached_system_prompts[query_task] = system_tokens_cache
    user_tokens = tokenizer(user_part, return_tensors="pt", add_special_tokens=False).input_ids
    full_tokens = torch.cat([system_tokens_cache, user_tokens], dim=-1).to(model.device)
    attention_mask = torch.ones_like(full_tokens)
    inputs = {
        "input_ids": full_tokens.to(model.device),
        "attention_mask": attention_mask.to(model.device)
    }

    
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # prompt_token_count = inputs.input_ids.shape[-1]

    # print(f'LLAMA prompt: \n{prompt}')
    prompt_len = inputs["input_ids"].shape[1]
    prompt_llama_tokens += prompt_len
    # print(f'len {prompt_len} \n prompt tokens {prompt_token_count}')
    with torch.no_grad():
        res = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=n,
        )
        # raw_output_text = tokenizer.decode(res[0], skip_special_tokens=True)
    for output in res:
        generated_tokens = output[prompt_len:]
        raw_output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        output_text = raw_output_text
            # print(f'Raw LLAMA output: {output_text}\n')
        # if prompt in output_text:
        #     output_text = output_text.replace(prompt, "")
        # if "assistant" in output_text:
        #     index = output_text.rfind("assistant")
        #     output_text =  output_text[(index + 9):]
        outputs.append(output_text)
        generated_token_count = output.shape[-1] - prompt_len
        llama_tokens += generated_token_count
            # print(f'LLAMA output: {output_text}\n')
        if(output_text == ""):
            print(f'LLAMA output empty, raw output: {raw_output_text}')
    # print(f"output: {outputs}")

    return outputs

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    # print(f"Messages: {messages}")
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

def reset():
    global completion_tokens, prompt_tokens, llama_tokens, prompt_llama_tokens
    completion_tokens = prompt_tokens = 0
    llama_tokens = prompt_llama_tokens = 0
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

def llama_usage():
    global prompt_llama_tokens, llama_tokens
    return {"llama_completion_tokens": llama_tokens, "llama_prompt_tokens": prompt_llama_tokens}