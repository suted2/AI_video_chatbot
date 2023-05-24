import librosa
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy import signal 
from librosa.filters import mel
from numpy.random import RandomState
import soundfile as sf
import numpy as np
from pysptk import sptk
import os
from tqdm import tqdm
import pickle


base = '/home/ubuntu/dataset/MIT/MIT_front_2d/'
txts = os.listdir(base)
data = []
def mk_float(n):
    return float(n)

import numpy as np

for txt in tqdm(sorted(txts)):
    temp = []
    if txt[-3:] == 'txt': 
        # print(txt)
        with open(base+txt, "r") as file:
            for i in file.readlines():
                strings = i.split('\n')[0]
                strings = strings.split(' ')[1:]

                # strings = strings.split(' ')

                temp.append(list(map(mk_float, strings)))
        temp = np.array(temp)
        # print(temp.shape)
# 여기서 shape_3d 를 temp로 바꾸면 될듯?

        # shape_3d, title = data  # shape_3d -> (batch, 204임)
#     # print(arr, title, sep='\n')
        shape_3d = temp.reshape([-1, 68, 3])
        # print(shape_3d.shape)

        scale = np.abs(1.0 / (shape_3d[:, 36:37, 0:1] - shape_3d[:, 45:46, 0:1]+10e-6))
        shift = - 0.5 * (shape_3d[:, 36:37] + shape_3d[:, 45:46])
        # print(f'scale : {scale.shape}, shift : {shift.shape}')
        shape_3d = (shape_3d + shift) * scale
        fl_data = (shape_3d.reshape(-1, 204), txt)
        # print(fl_data[0])
        # print(fl_data[1])

        data.append(fl_data) # a -> 텍스트 읽기
        # print(len(temp))
    # print(data)
    # break

with open('/home/ubuntu/MakeItTalk/dump_7/autovc_retrain_mel_std_train_fl.pickle', 'wb') as f:
    pickle.dump(data, f, pickle. HIGHEST_PROTOCOL)