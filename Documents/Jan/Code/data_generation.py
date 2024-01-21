import os.path

import pyttsx3

with open(r"D:\TaoLi\Projects\DialectTranslation\Documents\Jan\Code\result.txt", 'r', encoding='GBK') as f:
    data_input = f.readlines()
    sentences = []
    sentences_length = []
    for i in range(len(data_input)):
        if data_input[i][0] == '#':
            continue
        elif len(data_input[i].strip('\n')) <= 16:
            sentences.append(data_input[i].strip('\n'))
            sentences_length.append(len(data_input[i].strip('\n')))

engine = pyttsx3.init()

for i in range(len(sentences)):
    filename = f'{i+1}.wav'
    engine.save_to_file(sentences[i], os.path.join(r'D:\TaoLi\Projects\DialectTranslation\Documents\Jan\Code\Machine_Learning_Data\audio', filename))
    engine.runAndWait()
    print(f"第{i+1}个单词已经保存")
engine.stop()
