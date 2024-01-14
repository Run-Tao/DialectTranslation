from pydub import AudioSegment
import os

input_dir = "Data/mandarin/audio/preprocessed/nonsilence"
output_dir = "Data/mandarin/audio/preprocessed/timelength"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 23):
    input_path = f"{input_dir}/mandarin_{i}_nonsilence.wav"
    output_path = f"{output_dir}/mandarin_{i}_scaled_1.5s.wav"

    # 读取音频文件
    audio = AudioSegment.from_file(input_path, format="wav")

    # 计算当前持续时间和目标持续时间的比率

    current_duration = len(audio) / 1000  # 将毫秒转换为秒
    target_duration = 2.5
    if current_duration != 0:
        rate = target_duration / current_duration
        # 改变播放速率
        new_audio = audio.speedup(playback_speed=rate)

        # 保存更改后的音频文件
        new_audio.export(output_path, format="wav")
    else:
        print(f"Error: File {input_path} has zero duration.")

for i in range(22):
    audio = AudioSegment.from_file(input_path, format="wav")
    print(len(audio))