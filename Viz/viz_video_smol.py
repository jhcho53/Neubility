import torch
import cv2
import os
import time
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import textwrap

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 및 프로세서 불러오기
model_path = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2"
).to(DEVICE)

# 메시지 템플릿 설정
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Classify the image as either fight or general."}
        ]
    }
]

# 입력 및 출력 영상 경로 설정
input_video_path = "/home/jaehyeon/Desktop/neubility/Dataset/fight-20250331T062205Z-001/fight/31-1_cam01_fight03_place02_night_summer.mp4"
output_video_path = "output_annotated_video.mp4"
frame_interval = 30  # N프레임마다 한 번 추론

# 영상 불러오기
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {input_video_path}")

# 영상 정보
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 비디오 라이터 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 텍스트 줄바꿈 함수
def wrap_text(text, max_width_px, font_scale, thickness):
    wrapped_lines = []
    words = text.split()
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        text_size, _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if text_size[0] <= max_width_px:
            line = test_line
        else:
            wrapped_lines.append(line)
            line = word
    if line:
        wrapped_lines.append(line)
    return wrapped_lines

# 추론 시작
torch.cuda.synchronize()
start_time = time.time()

latest_output_text = "Loading..."

frame_idx = 0
with torch.inference_mode():
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # 이미지 추론
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt", do_resize=False)
            inputs = {k: (v.half() if v.dtype == torch.float32 else v).to(DEVICE) for k, v in inputs.items()}

            # 텍스트 생성
            output_ids = model.generate(**inputs, max_new_tokens=256)
            decoded_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            latest_output_text = decoded_text.strip()

        # 프레임 크기에 맞게 텍스트 시각화
        h, w, _ = frame.shape
        font_scale = max(h / 1280, 0.3)
        thickness = max(int(font_scale * 3), 1)
        max_text_width = int(w * 0.5)

        wrapped_lines = wrap_text(latest_output_text.replace("<|im_end|>", ""), max_text_width, font_scale, thickness)

        (text_width, text_height), _ = cv2.getTextSize("Test", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        line_spacing = int(text_height * 1.4)

        box_width = 0
        for line in wrapped_lines:
            (line_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            box_width = max(box_width, line_w)

        x = int(w * 0.03)
        y_start = int(h * 0.08)
        box_height = line_spacing * len(wrapped_lines) + 10

        # 반투명 배경 박스
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y_start - text_height - 10),
            (x + box_width + 10, y_start + box_height),
            (0, 0, 0),
            thickness=-1
        )
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 텍스트 출력
        for i, line in enumerate(wrapped_lines):
            y = y_start + i * line_spacing
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA
            )

        out.write(frame)
        frame_idx += 1

torch.cuda.synchronize()
end_time = time.time()

# 정리
cap.release()
out.release()

# 결과 출력
elapsed = end_time - start_time
print(f"\n✅ Output saved to {output_video_path}")
print(f"⏱ Total time: {elapsed:.2f}s, Processed {frame_idx} frames")
print(f"⚡ Effective FPS: {frame_idx / elapsed:.2f}")
