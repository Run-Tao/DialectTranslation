import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 读取音频文件
audio_file = 'C:\\Users\\Run Running\\Documents\\test.wav'
y, sr = librosa.load(audio_file)

# 生成声谱图
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# 将声谱图转换成分贝单位
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# 绘制声谱图
plt.figure(figsize=(10, 6))
librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr, hop_length=512, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()