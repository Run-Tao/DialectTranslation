import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
import re


# 创建自定义数据集类
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


# 创建多层感知机（MLP）模型
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = nn.Linear(hidden_size_3, hidden_size_4)
        self.fc5 = nn.Linear(hidden_size_4, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


# 数据集路径和标签文件
data_folder = 'D:/TaoLi/Projects/DialectTranslation/Documents/Jan/Code/Machine_Learning_Data/energy_data_AI'
label_file = 'D:/TaoLi/Projects/DialectTranslation/Documents/Jan/Code/result.txt'

# 创建数据集实例
custom_dataset = CustomDataset(data_folder, label_file)

# 模型超参数
input_size = 100  # 根据您的数据维度设置
hidden_size_1 = 200
hidden_size_2 = 144
hidden_size_3 = 80
hidden_size_4 = 64
output_size = 40  # 假设有10个类别

# 检查并设置是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建MLP模型实例，并将其移动到GPU
model = MLPModel(input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 数据加载器和训练循环
batch_size = 32
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 100
epoch = 0
flag = False

while flag == False:
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['data'].to(device), data['label'].to(device)  # 将数据移动到GPU

        optimizer.zero_grad()
        outputs = model(inputs.float())  # 不再需要添加额外的维度
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每100个mini-batches输出一次
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            if running_loss / 100 <= 1:
                flag = True
            running_loss = 0.0
            epoch = epoch + 1

print('Finished Training')

# 保存模型到指定路径
model_save_path = 'D:/TaoLi/Projects/DialectTranslation/Documents/Jan/Code/my_model.pth'
torch.save(model.state_dict(), model_save_path)

# 加载已训练好的模型并进行推理
model_test = MLPModel(input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size).to(device)
model_test.load_state_dict(torch.load(model_save_path))
model_test.eval()

# 准备测试数据，这里使用随机数据作为示例
test_data = np.random.rand(1, input_size)
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)

# 进行推理
with torch.no_grad():
    output = model_test(test_data)

# 输出推理结果
print(output)
