import numpy as np
from scipy import linalg
from cvxopt import solvers
from .Utils import ndarray2matrix
from .kernel import *

class SVM(object):
    def __init__(self, kernel ,C=np.inf, kernelname="linear"):
        self.C = C
        self.kernel = kernel
        self.name = 'Nonlinear soft margin C=%.2f,kernel=%s'%(C, kernelname)

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

        K = np.zeros([m, m])
        for i in range(m):
            for j in range(m):
                K[i, j] = self.kernel(X[i], X[j])

        # 优化目标 min 1/2 X.T*P*X + q.T*X,此处的np.outer和np.dot等价
        P = ndarray2matrix(np.outer(y, y) * K)
        q = ndarray2matrix(-np.ones([m, 1]))

        # 约束条件Gx <= h
        g1 = np.eye(m) * -1
        g2 = np.eye(m)
        h1 = np.zeros(m)
        h2 = np.ones(m) * self.C
        G = ndarray2matrix(np.vstack([g1, g2]))
        h = ndarray2matrix(np.hstack([h1, h2]))

        # 约束条件Ax = b
        A = ndarray2matrix(y.T)
        b = ndarray2matrix(np.zeros(1))

        # 通过求解器求解
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        # 找到支持向量
        epsilon = 1e-5
        #     index = np.logical_and(alphas > epsilon, alphas < C - epsilon).flatten()
        index = (alphas > epsilon).flatten()
        support_vector_num = np.sum(index == True)
        si = np.arange(len(alphas))[index]
        self.salphas = alphas[index, :]
        self.sx = X[index, :]
        self.sy = y[index, :]

        # 找到偏移量
        self.b = np.zeros(1)
        for i in range(support_vector_num):
            tmp = np.zeros(1)
            for j in range(support_vector_num):
                tmp += self.salphas[j] * self.sy[j] * K[si[j], si[i]]
            self.b += self.sy[i, :] - np.sign(tmp)
        self.b /= support_vector_num

    def predict(self, x):
        res = self.b.copy()
        x = x.flatten()
        for alpha_, sx_, sy_ in zip(self.salphas, self.sx, self.sy):
            res += alpha_ * sy_ * self.kernel(sx_, x)
        return np.sign(res).item()
