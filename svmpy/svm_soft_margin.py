import numpy as np
from cvxopt import solvers
from .Utils import ndarray2matrix

class SVM_soft_margin:
    def __init__(self, C = np.inf):
        self.C = C
        self.name = 'Linear soft margin C=%.2f'%(C)

    def train(self, X, y):
        """在给定输入数据和标签的情况下估计拉格朗日因子α，并求得斜率ω和偏移b
            Args:
                X (numpy.array): Data Matrix，输入数据
                y (numpy.array): Response Vector，标签数据
                C (Constant)：权衡因子
            Returns:
                ω(numpy.array): 斜率
                b (numpy.array)：偏移
         """
        # 获取维度信息
        [m, n] = X.shape

        # 优化目标 min 1/2 X.T*P*X + q.T*X等价于P_ = np.dot((y*X), (y*X).T)
        P = ndarray2matrix(np.multiply(np.dot(y,y.T), np.dot(X,X.T)))

    #     P = ndarray2matrix(np.dot((y*X), (y*X).T))
        q = ndarray2matrix(-np.ones([m, 1]))

        # 约束条件Gx <= h
        g1 = np.eye(m)*-1
        g2 = np.eye(m)
        h1 = np.zeros(m)
        h2 = np.ones(m)*self.C
        G = ndarray2matrix(np.vstack([g1, g2]))
        h = ndarray2matrix(np.hstack([h1, h2]))

        # 约束条件Ax = b
        A = ndarray2matrix(y.T)
        b = ndarray2matrix(np.zeros(1))

        # 通过求解器求解
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        # 根据拉格朗日因子求解ω和α
        self.w = np.dot((alphas * y).T, X).T;

        # 找到支持向量
        epsilon = 1e-4
        index = (alphas > epsilon).flatten()
        self.b = y - np.dot(X, self.w)
        self.b = sum(self.b[index]) / np.sum(index == True)

        # 记录支持向量
        self.sx = X[index,:]
        self.sy = y[index,:]

        return self.w, self.b

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b).item()