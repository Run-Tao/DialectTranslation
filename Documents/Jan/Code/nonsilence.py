from pydub import AudioSegment

num_files = 22

for i in range(1, num_files + 1):
    # 加载原始音频文件
    audio_path = f'Data\\mandarin\\audio\\origin\\录音 ({i}).wav'
    audio = AudioSegment.from_file(audio_path, format="wav")

    # 删除开头和结尾的静默部分，设置参数按照具体情况调整
    non_silent_audio = audio.strip_silence(silence_len=100, silence_thresh=-40)

    # 生成处理后的文件路径
    output_path = f'Data\\mandarin\\audio\\preprocessed\\nonsilence\\mandarin_{i}_nonsilence.wav'

    # 保存处理后的音频文件
    non_silent_audio.export(output_path, format="wav")
