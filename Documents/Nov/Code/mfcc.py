import librosa
import numpy as np
import pandas as pd
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs
mfcc_data = []
for i in range(1, 23):
    file_path = f'Data/mandarin/audio/录音 ({i}).wav'
    mfccs = extract_mfcc(file_path)
    mfcc_data.append(mfccs)
    mfccs_csv = pd.DataFrame(mfccs)
    mfccs_csv.to_csv(f'Data\mandarin\MFCCs\录音 ({i}).csv')
# print(mfcc_data)
