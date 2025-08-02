from text_generation import Client

client = Client("http://localhost:8080")  # or your TGI endpoint
prompt = (
    "<|start_header_id|>system<|end_header_id|>\n"
    "You are a helpful assistant.<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n"
    "What is the capital of France?<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

response = client.generate(prompt=prompt, return_full_text=False)
print(f'response: {response.generated_text}')  # Output: Paris
