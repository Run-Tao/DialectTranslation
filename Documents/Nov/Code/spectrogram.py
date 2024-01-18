import os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 设置音频所在文件夹和存储的文件夹路径
input_audio_folder = r'D:\TaoLi\Projects\DialectTranslation\Data\mandarin\audio\GeneratedByAI'
output_spectrogram_folder = r'D:\Taoli\Projects\DialectTranslation\Data\mandarin\spectrogram'

# 确保存储声谱图的文件夹存在
os.makedirs(output_spectrogram_folder, exist_ok=True)

# 遍历指定文件夹内的所有音频文件
for i in range(100):
    audio_file_path = os.path.join(input_audio_folder, f'mandarin_{i}_AI.wav')

    # 生成音频的声谱图
    y, sr = librosa.load(audio_file_path)
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(os.path.join(output_spectrogram_folder, f'mandarin_{i}_spectrogram.png'))
