import numpy as np

"""
N: 隐状态数目, 比如掷骰子，有三种骰子，那么N=3
M：观测状态数目，有三种筛子，可以掷出1~8，那么M=8
A： 状态转移矩阵 N*N，意思是从已知骰子换到下一个骰子的概率矩阵
B： 观测矩阵 N*M，有三种骰子，每种骰子分别产生1~8的概率矩阵
pi: 初始概率 1*N，初始选择某个骰子的概率
T： 观测序列长度，所观测到的掷出来的数字序列的长度
O： 观测序列，所观测到的掷出来的数字序列
"""


def viterbi(N, M, A, B, pi, T, O):
    delta = np.zeros((T, N))  # delta为局部概率，即达到某个特殊中间状态的概率
    psi = np.zeros((T, N))  # psi为反向指针，指向最优的引发当前状态的前一时刻的某个状态

    # 初始化，计算初始时刻所有状态的局部概率，反向指针均为0
    # delta[0] = pi * B[:, O[1] - 1]
    for i in range(N):
        delta[0][i] = pi[i] * B[i][O[0] - 1]
        psi[0][i] = 0

    for t in range(1, T):
        for i in range(N):
            # 计算t-1时刻每个状态局部概率与到t时刻第i个状态的转移概率之积
            val = delta[t-1] * A[:, i]
            # 从t-1时刻到t时刻第i个状态的最大概率, 以及状态序号
            maxval = max(val)
            maxval_ind = np.argmax(val)
            # t时刻第i个状态的局部概率
            delta[t][i] = maxval * B[i][O[t] - 1]
            # t时刻第i个状态的反向指针
            psi[t][i] = maxval_ind

    # 终止，观测序列的概率等于T时刻的局部概率
    P = max(delta[T-1])
    I = [0] * T
    # T时刻的隐藏状态
    I[T-1] = np.argmax(delta[T-1]) + 1

    # 最优路径回溯
    for t in range(T-2, -1, -1):
        I[t] = int(psi[t+1][I[t+1] - 1] + 1)

    return I, P, delta


M = 2 # 一共有红、白两种球
N = 3 # 有三个盒子
A = np.array([[0.5, 0.2, 0.3],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]])
pi = np.array([0.2, 0.4, 0.4])
T = 3
O = [1, 2, 1]



