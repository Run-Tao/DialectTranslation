import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import string
import re


class CustomDataset(Dataset):
    def __init__(self, data_folder, label_file, max_data_index=3278):
        self.data_folder = data_folder
        with open(label_file, 'r', encoding='gbk') as file:
            lines = file.readlines()
            lines = lines[:max_data_index]  # 仅保留最大索引为3278的数据
            self.labels = [len(self.remove_chinese_punctuation(line.strip())) for line in lines]

        self.data_files = [f"{data_folder}/{i}.npy" for i in range(1, max_data_index + 1)]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        label = self.labels[idx]
        sample = {'data': data, 'label': label}
        return sample

    def remove_chinese_punctuation(self, text):
        chinese_punctuations = "！？｡。，、；：「」『』“”‘’《》〈〉﹏（）【】－——·～"
        return re.sub(r"[%s]+" % chinese_punctuations, "", text)


# 创建一维卷积神经网络（1D CNN）模型
class CNN1DModel(nn.Module):
    def __init__(self, input_size, num_classes, new_length):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(16 * new_length, num_classes)  # 使用 new_length

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将数据转换成 (batch_size, input_length, 1) 的格式
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)  # 展开成(batch_size, 16 * new_length)
        x = self.fc(x)
        return x


# 数据集路径和标签文件
data_folder = 'D:/TaoLi/Projects/DialectTranslation/Documents/Jan/Code/Machine_Learning_Data/energy_data_AI'
label_file = 'D:/TaoLi/Projects/DialectTranslation/Documents/Jan/Code/result.txt'

# 创建数据集实例
custom_dataset = CustomDataset(data_folder, label_file)

# 定义模型超参数
input_size = 100  # 根据您的数据维度设置
num_classes = 40  # 假设有40个类别
new_length = 10  # 假设新长度为10

# 创建1D CNN 模型实例
model = CNN1DModel(input_size, num_classes, new_length)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据加载器和训练循环
batch_size = 32
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['data'], data['label']

        # 调整数据维度
        inputs = inputs.reshape(-1, input_size, 1)

        optimizer.zero_grad()
        outputs = model(inputs.float())  # 不再需要添加额外的维度
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每100个mini-batches输出一次
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

print('Finished Training')

# 保存模型到指定路径
model_save_path = 'D:/TaoLi/Projects/DialectTranslation/Documents/Jan/Code/my_1dcnn_model.pth'
torch.save(model.state_dict(), model_save_path)