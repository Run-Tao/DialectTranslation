import pyttsx3
import os
from DT_pkgs import Constant_Data

data_path = "D:\\Taoli\\Projects\\DialectTranslation\\Data\\"
with open(data_path+'order.txt', 'r', encoding='utf-8') as f:
    words = f.readlines()
    for i in range(Constant_Data.DataLength):
        words[i] = words[i].strip('\n')


save_folder = "mandarin\\audio\\GeneratedByAI"
os.makedirs(data_path+save_folder, exist_ok=True)  # 创建存储音频文件的文件夹

engine = pyttsx3.init()

for i, word in enumerate(words):
    filename = f"mandarin_{i}_AI.wav"
    filepath = os.path.join(data_path, save_folder, filename)

    engine.save_to_file(word, filepath)
    engine.runAndWait()

    print(f"单词 '{word}' 已保存到 {filepath}")

engine.stop()
