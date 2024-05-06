import os
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def approximate_matrix(A, k):
    """
    计算矩阵A的秩为k的近似矩阵。

    参数:
    A -- 原始矩阵 (numpy.ndarray)
    k -- 近似矩阵的秩 (int)

    返回:
    A_approx -- 近似矩阵 (numpy.ndarray)
    """
    # 对矩阵A进行奇异值分解
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # 保留前k个奇异值
    S_k = np.diag(S[:k])
    
    # 计算近似矩阵
    A_approx = U[:, :k] @ S_k @ Vt[:k, :]
    
    return A_approx


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
    
    A_k = approximate_matrix(mfccs, 6)

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