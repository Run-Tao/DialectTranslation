
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif'] = ['SimHei']
def generate_energy_distribution(audio_file, label):
    y, sr = librosa.load(audio_file)
    energy = librosa.feature.rms(y=y)
    return energy, label
energy_data = []
with open('Data/order.txt','r',encoding='utf-8') as f:
    labels = f.readlines()
    for i in range(len(labels)):
        labels[i] = labels[i].strip('\n')
for i in range(1, 23):
    audio_file = f'Data/mandarin/audio/preprocessed/timelength/mandarin_{i}_scaled_1.5s.wav'
    energy, label = generate_energy_distribution(audio_file, labels[i-1])
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
plt.savefig('Data/mandarin/energydistribution.png')
plt.show()