import os
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 指定输入和输出文件夹路径
input_folder = r'D:\TaoLi\Projects\DialectTranslation\Data\mandarin\audio\GeneratedByAI'
output_folder = r'D:\TaoLi\Projects\DialectTranslation\Data\mandarin\MFCC'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    

# 遍历输入文件夹中的每个文件
for i in range(100):
    file_name = f"mandarin_{i}_AI.wav"
    input_file_path = os.path.join(input_folder, file_name)

    # 读取音频文件
    data, sample_rate = librosa.load(input_file_path)

    # 计算MFCC
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=100)
    U, S, VT = np.linalg.svd(mfccs)
    # 选择保留的奇异值数量，即降维后的维度k
    k = 8

    # 构造降维后的矩阵
    # 我们只保留前k个奇异值，因此Sigma矩阵将是一个k x k的对角矩阵
    Sigma = np.diag(S[:k])

    # 降维后的矩阵A_k
    A_k = np.dot(U[:, :k], Sigma)

    print("降维后的矩阵A_k:")
    print(A_k)
    # print(len(mfccs))
    # 生成输出文件路径
    output_file_path = os.path.join(output_folder, f"mfcc_{i}.npy")

    # 保存MFCC到文件
    np.save(output_file_path, mfccs)
    output_file_path = os.path.join(output_folder, f'SVD\SVD{i}.npy')
    np.save(output_file_path,A_k)

    A_k_df = pd.DataFrame(A_k)
    A_k_df.to_csv(os.path.join(output_folder, f'SVD\SVD{i}.csv'), index=False,header=False)
    # print(f"MFCC for {file_name} saved to {output_file_path}")

    # 可视化MFCC并保存图片
    output_image_path = os.path.join(output_folder, f"SVD\SVD_{i}.png")
    plt.savefig(output_image_path)
    plt.close()
    # print(f"Visualization for {file_name} saved to {output_image_path}")
plt.show()