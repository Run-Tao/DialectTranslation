import pyttsx3

with open('Data/order.txt','r',encoding='utf-8') as f:
    words = f.readlines()
    for i in range(len(words))