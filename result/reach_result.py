import pickle
import numpy as np

reward_reach_ave_1 = []
reward_reach_time_1 = []
reward_reach_ave_4 = []
reward_reach_time_4 = []
reward_reach_ave_c = []
reward_reach_time_c = []

# 実験結果を読み込む
for i in range(1, 11):
    f = open('reach_result/reward_reach_time_1_' + str(i) + '.txt', 'rb')
    reward_reach_time_1.append(pickle.load(f))

    f = open('reach_result/reward_reach_time_4_' + str(i) + '.txt', 'rb')
    reward_reach_time_4.append(pickle.load(f))

    f = open('reach_result/reward_reach_time_c_' + str(i) + '.txt', 'rb')
    reward_reach_time_c.append(pickle.load(f))

# 第1四分位数，中央値，第3四分位数を計算
time_1_25, time_1_50, time_1_75 = np.percentile(reward_reach_time_1, [25, 50, 75], axis=0)

time_4_25, time_4_50, time_4_75 = np.percentile(reward_reach_time_4, [25, 50, 75], axis=0)

time_c_25, time_c_50, time_c_75 = np.percentile(reward_reach_time_c, [25, 50, 75], axis=0)

# 10回の試行の中央値と四分位範囲を表示
print('proposed:')
print(time_c_50)
print(time_c_75 - time_c_25)
print('all4d:')
print(time_4_50)
print(time_4_75 - time_4_25)
print('all1d:')
print(time_1_50)
print(time_1_75 - time_1_25)