import os.path
import librosa
import DT_pkgs
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyttsx3
import numpy as np

# 加载已训练好的模型
model_path = r'D:\TaoLi\Projects\DialectTranslation\Documents\Jan\Code\my_model.pth'
model = DT_pkgs.MLPModel(DT_pkgs.input_size, DT_pkgs.hidden_size, DT_pkgs.output_size)  # 使用与训练时相同的模型结构定义
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

data_length = []
predict_length = []
engine = pyttsx3.init()
with open(r'D:\TaoLi\Projects\DialectTranslation\Data\order.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].strip('\n')
        data_length.append(len(data[i]))

for i in range(DT_pkgs.DataLength):
    # 假设DT_pkgs.generate_energy_distribution返回numpy数组作为输入
    input_data = DT_pkgs.generate_energy_distribution(f'D:/TaoLi/Projects/DialectTranslation/Data/mandarin/audio/GeneratedByAI/mandarin_{i}_AI.wav')
    input_tensor = torch.from_numpy(input_data).float()  # 转换为PyTorch张量
    with torch.no_grad():
        output = model(input_tensor)
        predicted_length = torch.argmax(output).item()  # 假设输出是一个类别标签

    predict_length.append(predicted_length)

# 输出预测结果
print(predict_length)
