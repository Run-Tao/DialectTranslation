import os
from scipy.interpolate import interp1d
import librosa
import numpy as np
import DT_pkgs


AI_data_path = r'D:\TaoLi\Projects\DialectTranslation\Documents\Jan\Code\Machine_Learning_Data\audio'
AI_save_path = r'D:\TaoLi\Projects\DialectTranslation\Documents\Jan\Code\Machine_Learning_Data\energy_data_AI'
energy_distribution = []

for i in range(0, len(os.listdir(AI_data_path))):
    np.save(os.path.join(AI_save_path, f'{i+1}.npy'), DT_pkgs.generate_energy_distribution(os.path.join(AI_data_path, f'{i + 1}.wav'))[0])
    print(len(DT_pkgs.generate_energy_distribution(os.path.join(AI_data_path, f'{i + 1}.wav'))[0]))

