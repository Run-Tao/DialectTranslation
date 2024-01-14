import pyttsx3
import os

with open('Data/order.txt','r',encoding='utf-8') as f:
    words = f.readlines()
    for i in range(len(words)):
        words[i] = words[i].strip('\n')



save_folder = "Data\\mandarin\\audio\\ReadbyAI"
os.makedirs(save_folder, exist_ok=True)  # 创建存储音频文件的文件夹

engine = pyttsx3.init()

for i, word in enumerate(words):
    filename = f"mandarin_{i}_AI.wav"
    filepath = os.path.join(save_folder, filename)

    engine.save_to_file(word, filepath)
    engine.runAndWait()

    print(f"单词 '{word}' 已保存到 {filepath}")

engine.stop()
