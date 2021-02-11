import copy
import time
import numpy as np
import gym
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

env = gym.make("Acrobot-v1")

obs_num = env.observation_space.shape[0]
acts_num = env.action_space.n
HIDDEN_SIZE = 32


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(obs_num, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, acts_num)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = self.fc4(h)
        return y


STEP_NUM = 1000000  # trainの総ステップ数
MEMORY_SIZE = 10000  # メモリサイズ
BATCH_SIZE = 64  # バッチサイズ
EPSILON_DECREASE = 0.0001  # εの減少値
EPSILON_MIN = 0.1  # εの下限
START_REDUCE_EPSILON = 1000  # εを減少させるステップ数
TRAIN_FREQ = 10  # Q関数の学習間隔
UPDATE_TARGET_Q_FREQ = 20  # Q関数の更新間隔
GAMMA = 0.99  # 割引率
CHECK_FREQ = 20   # checkモードの間隔
CHECK_NUM = 20   # checkのエピソード数
LEARN_NUM = 10    # 繰り返す回数
RK = 1    # ルンゲクッタの近似次数の値

for learn_num in range(1, LEARN_NUM + 1):
    env.set_rk(RK)    # ルンゲクッタの近似次数を設定
    # モデル
    Q = NN()
    Q_ast = copy.deepcopy(Q)
    optimizer = optim.RMSprop(Q.parameters(), lr=0.00015, alpha=0.95, eps=0.01)

    total_step = 0  # 総ステップ数
    memory = []  # Replay Memory
    check_total_rewards = []  # check時の累積報酬記録用リスト

    check_time = 20   # checkモードに入るかどうかのしきい値
    EPSILON = 1.0  # ε-greedy法

    # 学習開始
    start = time.time()
    while total_step < STEP_NUM:

        # checkモード
        if time.time() - start > check_time:   # 一定時間経過していたらcheckモードに入る
            time_stop = time.time()
            env.set_rk(4)

            for check_epoch in range(CHECK_NUM):
                check_pobs = env.reset()  # 環境初期化
                check_done = False  # ゲーム終了フラグ
                check_total_reward = 0  # 累積報酬

                while not check_done:
                    # 最適な行動を予測
                    check_pobs_ = np.array(check_pobs, dtype="float32").reshape((1, obs_num))
                    check_pobs_ = Variable(torch.from_numpy(check_pobs_))
                    check_pact = Q(check_pobs_)
                    check_maxs, check_indices = torch.max(check_pact.data, 1)
                    check_pact = check_indices.numpy()[0]
                    # 行動
                    check_obs, check_reward, check_done, _ = env.step(check_pact)
                    check_total_reward += check_reward
                    check_pobs = check_obs
                check_total_rewards.append(check_total_reward)

            check_time += CHECK_FREQ
            env.set_rk(RK)
            start = start + (time.time() - time_stop)    # checkモードにかかった時間は含めない

        pobs = env.reset()  # 環境初期化
        done = False  # ゲーム終了フラグ
        total_reward = 0  # 累積報酬
        while not done:
            # 行動選択
            pact = env.action_space.sample()
            # ε-greedy法
            if np.random.rand() > EPSILON:
                # 最適な行動を予測
                pobs_ = np.array(pobs, dtype="float32").reshape((1, obs_num))
                pobs_ = Variable(torch.from_numpy(pobs_))
                pact = Q(pobs_)
                maxs, indices = torch.max(pact.data, 1)
                pact = indices.numpy()[0]
            # 行動
            obs, reward, done, _ = env.step(pact)
            # メモリに蓄積
            memory.append((pobs, pact, reward, obs, done))  # 状態、行動、報酬、行動後の状態、ゲーム終了フラグ
            if len(memory) > MEMORY_SIZE:  # メモリサイズを超えていれば消していく
                memory.pop(0)
            # 学習
            if len(memory) > BATCH_SIZE:  # バッチサイズ分溜まっていれば学習
                # 経験リプレイ
                if total_step % TRAIN_FREQ == 0:
                    memory_ = np.random.permutation(memory)

                    batch = np.array(memory_[0:BATCH_SIZE])  # 経験ミニバッチ
                    pobss = np.array(batch[:, 0].tolist(), dtype="float32").reshape((BATCH_SIZE, obs_num))
                    pacts = np.array(batch[:, 1].tolist(), dtype="int32")
                    rewards = np.array(batch[:, 2].tolist(), dtype="int32")
                    obss = np.array(batch[:, 3].tolist(), dtype="float32").reshape((BATCH_SIZE, obs_num))
                    dones = np.array(batch[:, 4].tolist(), dtype="bool")
                    # set y
                    pobss_ = Variable(torch.from_numpy(pobss))
                    q = Q(pobss_)
                    obss_ = Variable(torch.from_numpy(obss))
                    maxs, indices = torch.max(Q_ast(obss_).data, 1)
                    maxq = maxs.numpy()  # maxQ
                    target = copy.deepcopy(q.data.numpy())
                    for j in range(BATCH_SIZE):
                        target[j, pacts[j]] = rewards[j] + GAMMA * maxq[j] * (not dones[j])  # 教師信号
                    # Perform a gradient descent step
                    optimizer.zero_grad()
                    loss = nn.MSELoss()(q, Variable(torch.from_numpy(target)))
                    loss.backward()
                    optimizer.step()
                # target_Q関数の更新
                if total_step % UPDATE_TARGET_Q_FREQ == 0:
                    Q_ast = copy.deepcopy(Q)
            # εの減少
            if EPSILON > EPSILON_MIN and total_step > START_REDUCE_EPSILON:
                EPSILON -= EPSILON_DECREASE
            # 次の行動へ
            total_reward += reward
            total_step += 1
            pobs = obs

    # 結果の保存
    file_name = 'learn_check_total_rewards_all' + str(RK) + 'd_' + str(learn_num) + '.txt'
    f = open(file_name, 'wb')
    pickle.dump(check_total_rewards, f)

env.close()