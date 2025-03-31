import matplotlib.pyplot as plt
import numpy as np

# 모델 이름
model_names = ['ViLA1.5-2.7B','ViLA1.5-3B',  'UForm-500M', 'UForm-dpo', 'SmolVLM-256M']

# VRAM 사용량 (단위: GB), 각 모델의 [Base, Quantization] 순서
vram_usage = {
    'ViLA1.5-3B': [np.nan, 0.49],
    'ViLA1.5-2.7B': [0.33, 0.48],
    'UForm-500M': [np.nan, 1.2],
    'UForm-dpo': [np.nan, 1.2],
    'SmolVLM-256M': [np.nan, 0.25],
}

# OOM 표시 여부 (True인 막대에만 'OOM' 텍스트 표시됨)
oom_flags = {
    'ViLA1.5-3B': [True, False],
    'ViLA1.5-2.7B': [False, False],
    'UForm-500M': [True, False],
    'UForm-dpo': [True, False],
    'SmolVLM-256M': [True, False],
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

# 'OOM' 텍스트 표시
for idx, model in enumerate(model_names):
    # Base
    if oom_flags[model][0]:  # Base OOM
        plt.text(x[idx] - bar_width/2, 0.05, 'OOM',
                 ha='center', va='bottom', color='red', fontweight='bold', fontsize=20)
    else:  # 값 표시
        height = vram_base[idx]
        plt.text(x[idx] - bar_width/2, height + 0.01, f'{height:.2f} FPS',
                 ha='center', va='bottom', fontsize=16)

    # Quantized
    if oom_flags[model][1]:  # Quantized OOM
        plt.text(x[idx] + bar_width/2, 0.05, 'OOM',
                 ha='center', va='bottom', color='red', fontweight='bold', fontsize=20)
    else:  # 값 표시
        height = vram_quant[idx]
        plt.text(x[idx] + bar_width/2, height + 0.01, f'{height:.2f} FPS',
                 ha='center', va='bottom', fontsize=16)

# 폰트 설정 및 레이아웃
plt.rcParams.update({'font.size': 20})
plt.xticks(x, model_names, fontsize=20)
plt.xlabel('Model', fontsize=20)
plt.ylabel('VRAM Usage (GB)', fontsize=20)
plt.title('VRAM Usage: Base vs Quantized Models', fontsize=24)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
