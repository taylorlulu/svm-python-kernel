import numpy as np
from .Utils import quadprog

class SVM_original_space:
    def __init__(self):
        self.name = 'Linear original space'

    def train(self, x, y):
        """
        Inputs:
            x - 训练样本的数据
            y - 训练样本的标签数据
        Outputs:
            w - 硬分类分类面参数
            b - 硬分类分类面参数
        """
        [rows, cols] = x.shape
        # 构建优化目标min 1/2 x'*P*x+q'*x
        P = np.zeros([cols+1, cols+1])
        P[:cols,:cols] = np.eye(cols)
        q = np.zeros([cols+1, 1])

        # 构建约束条件Gx <= h
        G = np.hstack([np.multiply(-y,x),-y])
        h = -np.ones([rows, 1])

        # 添加约束Ax = b
        A = np.zeros([cols+1, cols+1])
        b = np.zeros([cols+1, 1])

        # 用求解器进行求解
        sol = quadprog(P, q, G, h)

        self.w = sol[0:cols,:]
        self.b = sol[cols,:]

        return self.w, self.b

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b).item()
