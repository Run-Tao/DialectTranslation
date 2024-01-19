import numpy as np
from Data import Constant_Data
import matplotlib.pyplot as plt
import pandas as pd
import os


def load_energy_distribution(path):
    return np.load(path)


'''def is_peak(data_array,label,difference_value):
    if'''


def find_peaks(data_array, gap):
    peak_counters = 0
    for k in range(len(data_array)):
        start = max(0,k-gap)
        if 1 <= k <= len(data_array) - 2 and data_array[k] >= data_array[k - 1] and data_array[k] >= data_array[k + 1]:
            peak_counters = peak_counters + 1
    return peak_counters


# test importing data from a py file
# print(Constant_Data.DataLength)

data_path = r"D:\TaoLi\Projects\DialectTranslation\Data"
order_path = os.path.join(data_path, "order.txt")
energy_data = []

with open(order_path, 'r', encoding='utf-8') as f:
    orders = f.readlines()
    for i in range(Constant_Data.DataLength):
        orders[i] = orders[i].strip('\n')
np_data_path = r"mandarin\energy_data"
for i in range(Constant_Data.DataLength):
    file_name = f"{orders[i]}_energy_distribution.npy"
    file_path = os.path.join(data_path, np_data_path, file_name)
    energy_data.append(load_energy_distribution(file_path))
print(energy_data[27][0])
print(orders[27])
# test find_peaks
print(find_peaks(energy_data[27][0]))
'''
for i in range(Constant_Data.DataLength):
    for i,j
'''
