import numpy as np
import librosa
from scipy.interpolate import interp1d
import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def resample_audio(audio_file, target_length):
    y = audio_file  # 加载音频文件
    current_length = len(y)

    # 创建线性插值函数
    x = np.arange(0, current_length)
    f = interp1d(x, y, kind='linear')

    # 生成新的时间点
    new_x = np.linspace(0, current_length - 1, target_length)

    # 对音频进行插值
    resampled_audio = f(new_x)

    return resampled_audio


def generate_energy_distribution(filename):
    y, sr = librosa.load(filename)
    energy_feature = librosa.feature.rms(y=resample_audio(y, 51000))
    return energy_feature
