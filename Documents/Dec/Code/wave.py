import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# 存储所有波形数据和对应的标题
all_waveforms = []
operation_names = ['开门','开灯','关门','关灯','开空调','关空调','一档风','二档风','三档风','静音','睡眠',
                   '制冷','制热','升温','降温','开电风扇','关电风扇','摇头','停止摇头','扫地机器人开始工作',
                   '开热水器','关热水器']

# 遍历处理每个音频文件
for i in range(1, 23):
    # 读取音频文件
    audio_path = f'Data\\mandarin\\audio\\preprocessed\\timelength\\mandarin_{i}_scaled_1.5s.wav'
    data, samplerate = sf.read(audio_path)

    # 计算时间轴
    time = np.arange(0, len(data)) / samplerate

    # 绘制波形图（使用深蓝色）
    plt.figure(figsize=(12, 8))
    plt.plot(time, data, color='#1f77b4')  # 设置波形图的颜色为深蓝色
    plt.title(operation_names[i-1])  # 设置正确的中文标题
    save_path = f'Data\\mandarin\\wavegraph\\waveform_{operation_names[i-1]}.png'
    plt.savefig(save_path)
    # 存储波形数据
    all_waveforms.append(data)

# 将所有波形图整合到一个大的图片文件中
plt.figure(figsize=(16, 10))
for i in range(22):
    plt.subplot(6, 4, i+1)
    plt.plot(np.arange(0, len(all_waveforms[i])) / samplerate, all_waveforms[i], color='#1f77b4')
    plt.title(operation_names[i])
plt.tight_layout()
plt.savefig('Data\\mandarin\\all_waveforms_subplot.png')

print("All waveform images combined into a single image using subplots.")
