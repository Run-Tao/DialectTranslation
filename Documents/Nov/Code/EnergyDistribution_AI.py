import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 设置音频所在文件夹和存储的文件夹路径
input_audio_folder = r'D:\TaoLi\Projects\DialectTranslation\Data\mandarin\audio\GeneratedByAI'
output_energy_distribution_folder = r'D:\Taoli\Projects\DialectTranslation\Data\mandarin\energy_distribution'

# 确保存储能量分布曲线的文件夹存在
os.makedirs(output_energy_distribution_folder, exist_ok=True)

# 遍历指定文件夹内的所有音频文件
for i in range(100):
    audio_file_path = os.path.join(input_audio_folder, f'mandarin_{i}_AI.wav')

    # 读取音频数据
    y, sr = librosa.load(audio_file_path)

    # 计算每帧的能量
    energy = librosa.feature.rms(y=y)

    # 画出能量分布曲线
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.4)
    plt.plot(energy.T, color='r')
    plt.title('Energy Distribution')
    plt.savefig(os.path.join(output_energy_distribution_folder, f'mandarin_{i}_energy_distribution.png'))
