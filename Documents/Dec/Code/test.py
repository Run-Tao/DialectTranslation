import librosa
import librosa.display
import matplotlib.pyplot as plt

# 加载音频文件
y, sr = librosa.load('Data/mandarin/录音 (17).wav')

# 计算能量
energy = librosa.feature.rms(y=y)

# 绘制能量分布图

librosa.display.specshow(energy, x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Energy Distribution')
plt.show()