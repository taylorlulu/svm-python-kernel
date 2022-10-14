import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

# 二维图像线性分割面绘图
def plot_svc_decision_boundary(model, x, y):
    if len(model.w) != 2:
        print("只适用于二维图像绘制")
        return

    index1 = (y == 1).flatten()
    index2 = (y == -1).flatten()

    plt.figure()
    plt.plot(x[:, 0][index1], x[:, 1][index1], "bo")
    plt.plot(x[:, 0][index2], x[:, 1][index2], "ms")

    w = model.w
    b = model.b

    xmin = np.min(x)
    xmax = np.max(x)

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    # 标注支持向量
    plt.scatter(model.sx[:, 0], model.sx[:, 1], s=180, facecolors='#FFAAAA')

    # 绘制分割直线
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

    # 其他设置
    plt.xlabel("$x'_0$", fontsize=15)
    plt.ylabel("$x'_1$  ", fontsize=15, rotation=0)
    plt.title("%s"%(model.name), fontsize=16)

# 二维图像线性或非线性分割面绘图
def plot_svc_decision_boundary2(svm, x, y, grid_size = 200):
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(svm.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z,
                 cmap=plt.cm.brg,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.2)
    plt.scatter(flatten(x[:, 0]), flatten(x[:, 1]),
                c=flatten(y),cmap=plt.cm.brg,alpha=0.8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel("$x'_0$", fontsize=15)
    plt.ylabel("$x'_1$  ", fontsize=15, rotation=0)
    plt.title("%s" % (svm.name), fontsize=16)
