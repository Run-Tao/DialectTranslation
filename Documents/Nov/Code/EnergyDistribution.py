# Using data read by myself
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import numpy as np
from DT_pkgs import Constant_Data

# matplotlib for chinese
plt.rcParams['font.sans-serif'] = ['SimHei']


def generate_energy_distribution(filename, label_name):
    y, sr = librosa.load(filename)
    energy_feature = librosa.feature.rms(y=y)
    return energy_feature, label_name


energy_data = []
data_path = r"D:/TaoLi/Projects/DialectTranslation/Data"
with open(data_path+'/order.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    for i in range(len(labels)):
        labels[i] = labels[i].strip('\n')
for i in range(Constant_Data.DataLength):
    audio_file = data_path + f'/mandarin/audio/GeneratedByAI/mandarin_{i}_AI.wav'
    energy, label = generate_energy_distribution(audio_file, labels[i - 1])
    energy_data.append((energy, label))
num_images = len(energy_data)
num_cols = 5
num_rows = math.ceil(num_images / num_cols)
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))
for i in range(num_images):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.plot(energy_data[i][0][0])
    ax.set_title(f'{energy_data[i][1]}')
for i in range(num_images, num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])
plt.tight_layout(rect=[0, 0.01, 1, 0.95])
plt.suptitle('能量分布')
plt.savefig(data_path+'/mandarin/energy_distribution_AI.png')
plt.show()

save_path = os.path.join(data_path, 'mandarin', 'energy_data')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 保存energy_data
for (energy, label) in energy_data:
    save_file = os.path.join(save_path, f'{label}_energy_distribution.npy')
    np.save(save_file, energy)
