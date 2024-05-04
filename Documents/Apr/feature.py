import math
import librosa
import numpy
import os

mfcc_path = f'D:\TaoLi\Projects\DialectTranslation\Data\mandarin\MFCC'
for i in range(100):
    file_name = os.path.join(mfcc_path, f"mfcc_{i}.npy")
    mfcc = numpy.load(file_name)
    
    