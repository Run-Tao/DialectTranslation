import numpy as np
from Data import Constant_Data
import matplotlib.pyplot as plt
import os
from collections import Counter


# This function is used to load the data of energy distribution which was calculated in another program.
def load_energy_distribution(path):
    return np.load(path)


# This function is used to find out how many peaks there are in the energy distribution curve of an audio clip. And the result will be used in calculating how much words are there in a certain audio clip.
# Why gap == 3 ? Because of the accuracy. When gap == 3 or 4 , the accuracy is 85%; when gap == 2, the accuracy is 81%. And when gap is 4, the complete accuracy is lower than the rate when gap == 3.
def find_peaks(data_array, gap=2):
    peak_counters = 0
    peak_array = []
    for k in range(len(data_array)):
        start = max(0, k - gap)
        end = min(len(data_array), k + gap)
        if data_array[k] == max(data_array[start:end]) and data_array[k] != 0 and data_array[k] != min(data_array[start:end]):
            peak_counters += 1
            peak_array.append(data_array[k])
    return peak_counters, peak_array


# test importing data from a py file
# print(Constant_Data.DataLength)


order_path = os.path.join(Constant_Data.data_path, "order.txt")
energy_data = []

with open(order_path, 'r', encoding='utf-8') as f:
    orders = f.readlines()
    for i in range(Constant_Data.DataLength):
        orders[i] = orders[i].strip('\n')
np_data_path = r"mandarin\energy_data"
for i in range(Constant_Data.DataLength):
    file_name = f"{orders[i]}_energy_distribution.npy"
    file_path = os.path.join(Constant_Data.data_path, np_data_path, file_name)
    energy_data.append(load_energy_distribution(file_path))
for i in range(Constant_Data.DataLength):
    print(orders[i], find_peaks(energy_data[i][0])[0], len(orders[i]))
orders_length = [len(orders[i]) for i in range(Constant_Data.DataLength)]
predict_length = [find_peaks(energy_data[i][0])[0] for i in range(Constant_Data.DataLength)]

errors = [abs(true - pred) for true, pred in zip(orders_length, predict_length)]

x = np.arange(Constant_Data.DataLength)

# 创建误差图
plt.errorbar(x, orders_length, yerr=errors, fmt='o', color='b', ecolor='r', linestyle='-', linewidth=2, capsize=5, capthick=2)
print(errors)
# 显示图形
plt.show()
plt.scatter(errors, x)
counts = Counter(errors)
# 打印结果
cnt_3 = counts.most_common(3)
cnt = 0
print(cnt_3)
for i in range(len(cnt_3)):
    cnt += cnt_3[i][1]
print(cnt/100)
for num, count in counts.items():
    print(f"{num} 出现了 {count} 次")
plt.show()
