import matplotlib.pyplot as plt
import numpy as np

# 모델 이름
model_names = ['ViLA1.5-2.7B', 'ViLA1.5-3B', 'UForm-500M', 'UForm-dpo', 'SmolVLM-256M', 'SmolVLM-256M-Video']

# VRAM 사용량 (단위: GB), 각 모델의 [Base, Quantization] 순서
vram_usage = {
    'ViLA1.5-3B': [4.4, 3.6],
    'ViLA1.5-2.7B': [4.0, 3.1],
    'UForm-500M': [5.4, 2.9],
    'UForm-dpo': [5.4, 2.9],
    'SmolVLM-256M': [6.4, 2.5],
    'SmolVLM-256M-Video': [5.5, 2.6]
}

# OOM 표시 여부 (True인 막대에만 'OOM' 텍스트 표시됨)
oom_flags = {
    'ViLA1.5-3B': [True, False],
    'ViLA1.5-2.7B': [False, False],
    'UForm-500M': [True, False],
    'UForm-dpo': [True, False],
    'SmolVLM-256M': [True, False],
    'SmolVLM-256M-Video': [True, False]
}

# 막대 설정
x = np.arange(len(model_names))
bar_width = 0.35

# 데이터 정리
vram_base = [vram_usage[m][0] for m in model_names]
vram_quant = [vram_usage[m][1] for m in model_names]

# 그래프 그리기
plt.figure(figsize=(20, 10))
bars_base = plt.bar(x - bar_width/2, vram_base, width=bar_width, label='Base')
bars_quant = plt.bar(x + bar_width/2, vram_quant, width=bar_width, label='Quantized')

# OOM 또는 값 텍스트 출력
for idx, model in enumerate(model_names):
    if oom_flags[model][0]:  # Base OOM
        height = vram_base[idx]
        plt.text(x[idx] - bar_width/2, height / 2, 'OOM',
                 ha='center', va='center', color='white', fontweight='bold', fontsize=20)
    else:
        height = vram_base[idx]
        plt.text(x[idx] - bar_width/2, height + 0.1, f'{height:.2f} GB',
                 ha='center', va='bottom', fontsize=16)

    if oom_flags[model][1]:  # Quantized OOM
        height = vram_quant[idx]
        plt.text(x[idx] + bar_width/2, height / 2, 'OOM',
                 ha='center', va='center', color='white', fontweight='bold', fontsize=20)
    else:
        height = vram_quant[idx]
        plt.text(x[idx] + bar_width/2, height + 0.1, f'{height:.2f} GB',
                 ha='center', va='bottom', fontsize=16)

# 시각 설정
plt.rcParams.update({'font.size': 20})
plt.xticks(x, model_names, fontsize=20)
plt.xlabel('Model', fontsize=20)
plt.ylabel('VRAM Usage (GB)', fontsize=20)
plt.title('VRAM Usage: Base vs Quantized Models')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
