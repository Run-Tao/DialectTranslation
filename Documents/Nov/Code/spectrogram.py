import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 读取音频文件并生成声谱图
def generate_spectrogram(audio_file):
    y, sr = librosa.load(audio_file)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(spectrogram, ref=np.max), sr

# 创建一个包含所有声谱图的列表
spectrograms = []
srs = []
for i in range(1, 23):
    audio_file = f'Data\\mandarin\\audio\\preprocessed\\timelength\\mandarin_{i}_scaled_1.5s.wav'
    spectrogram, sr = generate_spectrogram(audio_file)
    spectrograms.append(spectrogram)
    srs.append(sr)

# 将所有声谱图整合到一个输出图片中
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))

with open('Data/order.txt','r',encoding='utf-8') as f:
    title = f.readlines()
    for i in range(len(title)):
        title[i] = title[i].strip('\n')

for i in range(22):
    row = i // 5
    col = i % 5
    ax = axes[row, col]
    librosa.display.specshow(spectrograms[i], sr=srs[i], x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title[i])
    
num_images = 22
num_rows = 5
num_cols = 5
for i in range(num_images, num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout(rect=[0, 0.01, 1, 0.95])
plt.suptitle('普通话智能家居语音指令音频声谱图')
plt.savefig('Data\mandarin\spectrogram.png')
plt.show()


