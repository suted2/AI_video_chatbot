import pickle
import numpy as np

# with open('/home/ubuntu/MakeItTalk/dump_4/autovc_retrain_mel_val_au.pickle', 'rb') as fp:
#     au_data = pickle.load(fp)
# au_mean_std = np.loadtxt('/home/ubuntu/MakeItTalk/src/dataset/utils/MEAN_STD_AUTOVC_RETRAIN_MEL_AU.txt')
# au_mean, au_std = au_mean_std[0:au_mean_std.shape[0]//2], au_mean_std[au_mean_std.shape[0]//2:]

# au_data = [(au - au_mean) / au_std for au in au_data]
# with open('/home/ubuntu/MakeItTalk/dump_4/autovc_retrain_mel_std_val_au.pickle', 'wb') as fp:
#     pickle.dump(au_data, fp)
# print(len(au_data))

# train
# with open('/home/ubuntu/MakeItTalk/dump_5/autovc_retrain_mel_train_au_matchLength.pickle', 'rb') as fp:
#     au_data = pickle.load(fp)
# au_mean_std = np.loadtxt('/home/ubuntu/MakeItTalk/src/dataset/utils/MEAN_STD_AUTOVC_RETRAIN_MEL_AU.txt')
# au_mean, au_std = au_mean_std[0:au_mean_std.shape[0]//2], au_mean_std[au_mean_std.shape[0]//2:]

# au_data = [(au - au_mean) / au_std for au in au_data]
# with open('/home/ubuntu/MakeItTalk/dump_5/autovc_retrain_mel_std_train_au.pickle', 'wb') as fp:
#     pickle.dump(au_data, fp)
# print(len(au_data))

# # valid
# with open('/home/ubuntu/MakeItTalk/dump_5/autovc_retrain_mel_val_au_matchLength.pickle', 'rb') as fp:
#     au_data = pickle.load(fp)
# au_mean_std = np.loadtxt('/home/ubuntu/MakeItTalk/src/dataset/utils/MEAN_STD_AUTOVC_RETRAIN_MEL_AU.txt')
# au_mean, au_std = au_mean_std[0:au_mean_std.shape[0]//2], au_mean_std[au_mean_std.shape[0]//2:]

# au_data = [(au - au_mean) / au_std for au in au_data]
# with open('/home/ubuntu/MakeItTalk/dump_5/autovc_retrain_mel_std_val_au.pickle', 'wb') as fp:
#     pickle.dump(au_data, fp)
# print(len(au_data))

# train6
with open('/home/ubuntu/MakeItTalk/dump_6/autovc_retrain_mel_train_au_matchLength.pickle', 'rb') as fp:
    au_data = pickle.load(fp)
au_mean_std = np.loadtxt('/home/ubuntu/MakeItTalk/src/dataset/utils/MEAN_STD_AUTOVC_RETRAIN_MEL_AU.txt')
au_mean, au_std = au_mean_std[0:au_mean_std.shape[0]//2], au_mean_std[au_mean_std.shape[0]//2:]

au_data = [(au - au_mean) / au_std for au in au_data]
with open('/home/ubuntu/MakeItTalk/dump_6/autovc_retrain_mel_std_train_au.pickle', 'wb') as fp:
    pickle.dump(au_data, fp)
print(len(au_data))