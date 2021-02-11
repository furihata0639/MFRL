import pickle
import matplotlib.pyplot as plt
import numpy as np


check_total_rewards_all1d = []
check_total_rewards_all4d = []
check_total_rewards_change = []

# 実験結果を読み込む
for i in range(1, 11):
    file_name = 'graph_result/learn_check_total_rewards_change_' + str(i) + '.txt'
    f = open(file_name, 'rb')
    tmp1 = pickle.load(f)
    # 1回のcheckが20エピソードなので，20エピソード毎の平均を計算する
    resize = (len(tmp1) // 20, 20)
    tmp2 = np.array(tmp1, dtype='float32').reshape(resize)
    check_total_rewards_change.append(np.average(tmp2, axis=1))
    # 試行毎にcheckの回数が微妙に違うので，一番少ないものに数を合わせる
    check_total_rewards_change[i - 1] = check_total_rewards_change[i - 1][:67]

    file_name = 'graph_result/learn_check_total_rewards_all4d_' + str(i) + '.txt'
    f = open(file_name, 'rb')
    tmp1 = pickle.load(f)
    resize = (len(tmp1) // 20, 20)
    tmp2 = np.array(tmp1, dtype='float32').reshape(resize)
    check_total_rewards_all4d.append(np.average(tmp2, axis=1))
    check_total_rewards_all4d[i - 1] = check_total_rewards_all4d[i - 1][:69]

    file_name = 'graph_result/learn_check_total_rewards_all1d_' + str(i) + '.txt'
    f = open(file_name, 'rb')
    tmp1 = pickle.load(f)
    resize = (len(tmp1) // 20, 20)
    tmp2 = np.array(tmp1, dtype='float32').reshape(resize)
    check_total_rewards_all1d.append(np.average(tmp2, axis=1))
    check_total_rewards_all1d[i - 1] = check_total_rewards_all1d[i - 1][:65]

# 第1四分位数，中央値，第3四分位数を計算
check_c_25, check_c_50, check_c_75 = np.percentile(check_total_rewards_change, [25, 50, 75], axis=0)
check_4_25, check_4_50, check_4_75 = np.percentile(check_total_rewards_all4d, [25, 50, 75], axis=0)
check_1_25, check_1_50, check_1_75 = np.percentile(check_total_rewards_all1d, [25, 50, 75], axis=0)

# 10試行の中央値と四分位範囲を描画
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('time:[sec]', fontsize=15)
ax.set_ylabel('reward', fontsize=15)

ax.plot(range(20, (20 * len(check_4_50) + 20), 20), check_4_50, label='fix(4d)', color='b', marker='o')
ax.fill_between(range(20, (20 * len(check_4_50) + 20), 20), check_4_25, check_4_75,
                 alpha=0.15, color='m')
ax.plot(range(20, (20 * len(check_1_50) + 20), 20), check_1_50, label='fix(1d)', color='r', marker='x')
ax.fill_between(range(20, (20 * len(check_1_50) + 20), 20), check_1_25, check_1_75,
                 alpha=0.15, color='r')
ax.plot(range(20, (20 * len(check_c_50) + 20), 20), check_c_50, label='proposed', color='g', marker='^')
ax.fill_between(range(20, (20 * len(check_c_50) + 20), 20), check_c_25,
                 check_c_75, alpha=0.15, color='g')
ax.legend(bbox_to_anchor=(1.1, 0.2), loc='upper right', borderaxespad=0)

sub_ax = plt.axes([0.37, 0.2, 0.38, 0.4])
sub_ax.set_xlim(0, 300)
sub_ax.set_ylim(-300, -50)
sub_ax.plot(range(20, (20 * len(check_4_50) + 20), 20), check_4_50, label='fix(4d)', color='b', marker='o')
sub_ax.fill_between(range(20, (20 * len(check_4_50) + 20), 20), check_4_25, check_4_75,
                 alpha=0.15, color='m')
sub_ax.plot(range(20, (20 * len(check_1_50) + 20), 20), check_1_50, label='fix(1d)', color='r', marker='x')
sub_ax.fill_between(range(20, (20 * len(check_1_50) + 20), 20), check_1_25, check_1_75,
                 alpha=0.15, color='r')
sub_ax.plot(range(20, (20 * len(check_c_50) + 20), 20), check_c_50, label='proposed', color='g', marker='^')
sub_ax.fill_between(range(20, (20 * len(check_c_50) + 20), 20), check_c_25,
                 check_c_75, alpha=0.15, color='g')


plt.show()
fig.savefig("graph_result.jpg")