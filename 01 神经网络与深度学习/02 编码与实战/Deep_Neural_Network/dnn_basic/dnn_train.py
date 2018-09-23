import numpy as np
from .utils import *

def compute_cost(AL, Y, parameters, lambd = 0):
    """
    实施等式（4）定义的成本函数。

    参数：
        AL - 与标签预测相对应的概率向量，维度为（1，示例数量）
        Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

    返回：
        cost - 交叉熵成本
    """
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m

    if lambd != 0:
        L2_regularization_cost = 0;
        L = len(parameters) // 2 #整除
        for l in range(L):
            L2_regularization_cost = L2_regularization_cost + np.sum(np.square(parameters["W" + str(l + 1)]))
        L2_regularization_cost = lambd * L2_regularization_cost / (2 * m)

        cost = cost + L2_regularization_cost

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost



def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数

    参数：
     parameters - 包含你的参数的字典
     grads - 包含梯度值的字典，是L_model_backward的输出

    返回：
     parameters - 包含更新参数的字典
                   参数[“W”+ str（l）] = ...
                   参数[“b”+ str（l）] = ...
    """
    L = len(parameters) // 2 #整除
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters







