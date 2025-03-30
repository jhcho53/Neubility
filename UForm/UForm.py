

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel
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
    "/home/jaehyeon/Desktop/neubility/Dataset/task1",
    transform=transform,
    frame_interval=30
)

y_true = []
y_pred = []

torch.cuda.synchronize()
start_time = time.time()
# Initialize processor and model
model = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)


prompt = "Analyze the image and determine which category it falls under from the following options: Fire, Smoke, Fallen Person, Fight, Weapon Being Carried, None of the Above. Provide only one category as your final answer."

for i in tqdm(range(len(dataset))):
    frame, label = dataset[i]
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    with torch.inference_mode():
     output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=256,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]

    predicted_label = extract_label(decoded_text)
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