from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import PIL.Image
from dataloader.dataloader import FightEventDataset


def pil_collate_fn(batch):
    images, labels = zip(*batch)  
    return list(images), list(labels)

# 이미지 전처리
def cv2_to_pil(image):
    return PIL.Image.fromarray(image)

# 데이터셋 경로와 전처리 지정
dataset = FightEventDataset(
    "/home/jaehyeon/Desktop/neubility/Dataset/task1",
    transform=cv2_to_pil
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=pil_collate_fn
)

# 모델 설정
from transformers import AutoConfig, AutoModel

model_path = "Efficient-Large-Model/NVILA-Lite-2B-hf-preview"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.float16).eval().cuda()  # CUDA 선택 시

# 라벨링 정의
label_prompt = "Analyze the image and determine which category it falls under from the following options: fire, smoke, fallen person, fight, weapon being carried, general. Provide only one category as your final answer in English."

results = []

for images, gt_labels in tqdm(dataloader):
    image = images[0]
    gt_label = gt_labels[0]

    response = model.generate_content([image, label_prompt])
    pred_label = response.lower().strip()

    print(f"GT: {gt_label}, Pred: {pred_label}")
    results.append((gt_label, pred_label))
