import librosa
import noisereduce as nr

# 加载音频文件
audio_data, _ = librosa.load('Data', sr=None)

# 应用基本的去噪处理
reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=audio_data, verbose=False)

# 保存处理后的音频文件
librosa.output.write_wav('output_audio_denoised.wav', reduced_noise, _)
