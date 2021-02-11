import pickle
import matplotlib.pyplot as plt
import numpy as np

# 実験結果を読み込む
f = open('scatter_result/times_1.txt', 'rb')
times = pickle.load(f)
f = open('scatter_result/total_rewards_1.txt', 'rb')
total_rewards = pickle.load(f)
f = open('scatter_result/rk_stack_1.txt', 'rb')
rk_stack = pickle.load(f)

# 結果の描画
fig = plt.figure()
ax = fig.add_subplot()
for i in range(len(times)):
    if rk_stack[i] == 1:
        if rk_stack[i+1] == 2:
            ax.scatter(times[i], total_rewards[i], c='r', label='1d')
        else:
            ax.scatter(times[i], total_rewards[i], c='r')
    elif rk_stack[i] == 2:
        if rk_stack[i+1] == 3:
            ax.scatter(times[i], total_rewards[i], c='b', label='2d')
        else:
            ax.scatter(times[i], total_rewards[i], c='b')
    elif rk_stack[i] == 3:
        if rk_stack[i+1] == 4:
            ax.scatter(times[i], total_rewards[i], c='g', label='3d')
        else:
            ax.scatter(times[i], total_rewards[i], c='g')
# 中央値の描画
a = len(times) // 20
for i in range(a):
    each_total_rewards = []
    for j in range(20):
        each_total_rewards.append(total_rewards[20*i+j])
    if rk_stack[i*20] != 4:
        if i == 1:
            ax.scatter(times[i * 20], np.median(each_total_rewards), c='k', label='median')
        else:
            ax.scatter(times[i * 20], np.median(each_total_rewards), c='k')

ax.set_xlabel('time:[sec]', fontsize=15)
ax.set_ylabel('reward', fontsize=15)

plt.legend(loc='lower right', fontsize=15)

plt.show()
fig.savefig("scatter_result.jpg")