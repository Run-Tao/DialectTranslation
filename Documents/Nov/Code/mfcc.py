import os
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt

# 指定输入和输出文件夹路径
input_folder = r'D:\TaoLi\Projects\DialectTranslation\Data\mandarin\audio\GeneratedByAI'
output_folder = r'D:\TaoLi\Projects\DialectTranslation\Data\mandarin\MFCC'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的每个文件
for i in range(100):
    file_name = f"mandarin_{i}_AI.wav"
    input_file_path = os.path.join(input_folder, file_name)

    # 读取音频文件
    data, sample_rate = librosa.load(input_file_path)

    # 计算MFCC
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)

    # 生成输出文件路径
    output_file_path = os.path.join(output_folder, f"mfcc_{i}.npy")

    # 保存MFCC到文件
    # np.save(output_file_path, mfccs)

    print(f"MFCC for {file_name} saved to {output_file_path}")

    # 可视化MFCC并保存图片
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC for {file_name}')
    plt.tight_layout()
    output_image_path = os.path.join(output_folder, f"mfcc_{i}.png")
    plt.savefig(output_image_path)
    plt.close()
    print(f"Visualization for {file_name} saved to {output_image_path}")
