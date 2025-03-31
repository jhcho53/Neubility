from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# Initialize NVML
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0번 사용
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path) 
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2"
).to("cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/home/jaehyeon/Desktop/neubility/Dataset/fire_1451.mp4"},
            {"type": "text", "text": "Analyze the image and determine which category it falls under from the following options: Fire, Smoke, Fallen Person, Fight, Weapon possession, General."}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

inputs = {k: v.to(DEVICE).half() if v.dtype == torch.float32 else v.to(DEVICE) for k, v in inputs.items()}

torch.cuda.synchronize()
start_time = time.time()

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
torch.cuda.synchronize()
end_time = time.time()
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

# GPU Memory Usage
mem_info = nvmlDeviceGetMemoryInfo(handle)
used_mb = mem_info.used // 1024**2
total_mb = mem_info.total // 1024**2

print(generated_texts[0])
print(f"Inference time: {end_time - start_time:.2f} seconds")
print(f"GPU Memory Usage: {used_mb} MB / {total_mb} MB")