

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, AutoModelForImageTextToText
from transformers.image_utils import load_image
from torchvision import transforms
from dataloader.dataloader import FightEventDataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 라벨 후보
possible_labels = [
    "fire",
    "smoke",
    "fallen person",
    "fight",
    "weapon being carried",
    "general",
]

# 텍스트에서 라벨 추출 함수
import re

def extract_label(text):
    text = text.lower()
    match = re.search(r'assistant:\s*(.+)', text)
    if match:
        response = match.group(1).strip().strip(".")
    else:
        response = text

    for label in possible_labels:
        if label in response:
            return label
    return "unknown"



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToPILImage(),
])

dataset = FightEventDataset(
    "/home/jaehyeon/Desktop/neubility/Dataset/weapon possession-20250331T062204Z-001/weapon possession",
    transform=transform,
    frame_interval=30
)

y_true = []
y_pred = []

torch.cuda.synchronize()
start_time = time.time()
# Initialize processor and model
model_path = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2"
).to("cuda")

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image."}
        ]
    },
]

for i in tqdm(range(len(dataset))):
    frame, label = dataset[i]
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt", do_resize=False)

    # float16으로 명시적 변환
    inputs = {k: v.to(DEVICE).half() if v.dtype == torch.float32 else v.to(DEVICE) for k, v in inputs.items()}

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    predicted_label = extract_label(generated_texts[0])
    y_true.append(label)
    y_pred.append(predicted_label)

torch.cuda.synchronize()
end_time = time.time()

# 5. 성능 출력
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, labels=possible_labels, zero_division=0))

# 6. Confusion Matrix 시각화
cm = confusion_matrix(y_true, y_pred, labels=possible_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=possible_labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 7. FPS 계산
elapsed = end_time - start_time
fps = len(dataset) / elapsed
print(f"\n⏱ Total inference time: {elapsed:.2f} seconds for {len(dataset)} frames")
print(f"⚡ Model FPS: {fps:.2f} frames/sec")