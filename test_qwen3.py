from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# pick base (not instruct) for raw completion
MODEL = "Qwen/Qwen3-8B-Base"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# load model (fp16 to save memory, auto-sharded if multiple GPUs)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# a simple test prompt
prompt = "Solve the following math problem:\nWhat is 12 * 13?\n\nFinal Answer:"

# tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# generate
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# decode
print(tokenizer.decode(outputs[0], skip_special_tokens=True))