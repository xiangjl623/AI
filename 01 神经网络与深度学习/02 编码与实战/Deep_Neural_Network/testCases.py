#testCase.py
import numpy as np
from dnn_basic.dnn_train import *
from dnn_basic.initialize import *

def initialize_parameters_deep_test_case():
    layers_dims = [5, 4, 3]
    parameters = initialize_parameters_deep(layers_dims)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

def linear_forward_test_case():
    np.random.seed(1)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    return A, W, b

def linear_activation_forward_test_case():
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

def L_model_forward_test_case():
    np.random.seed(1)
    X = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return X, parameters

def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8,.9,0.4]])

    return Y, aL

def linear_backward_test_case():
    np.random.seed(1)
    dZ = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    linear_cache = (A, W, b)
    return dZ, linear_cache

def linear_activation_backward_test_case():
    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    return dA, linear_activation_cache

def L_model_backward_test_case():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ( (A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches

def update_parameters_test_case():
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return parameters, grads

def test_case_run():
    #测试initialize_parameters_deep
    print("==============0 测试initialize_parameters_deep==============")
    initialize_parameters_deep_test_case()
    #测试linear_forward
    print("==============1 测试linear_forward==============")
    A,W,b = linear_forward_test_case()
    Z, linear_cache = linear_forward(A, W, b)
    print("Z = " + str(Z))
    #测试linear_activation_forward
    print("==============2 测试linear_activation_forward==============")
    A_prev, W,b = linear_activation_forward_test_case()
    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
    print("sigmoid，A = " + str(A))
    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
    print("ReLU，A = " + str(A))
    #测试L_model_forward
    print("==============3 测试L_model_forward==============")
    X, parameters = L_model_forward_test_case()
    AL, caches = L_model_forward(X, parameters)
    print("AL = " + str(AL))
    print("caches 的长度为 = " + str(len(caches)))
    print(caches[0][0])
    print(caches[0][1])
    print(caches[1][0])
    print(caches[1][1])
    #测试compute_cost
    print("==============测试compute_cost==============")
    Y, AL = compute_cost_test_case()
    print("cost = " + str(compute_cost(AL, Y)))
    #测试linear_backward
    print("==============测试linear_backward==============")
    dZ, linear_cache = linear_backward_test_case()
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    #测试linear_activation_backward
    print("==============测试linear_activation_backward==============")
    AL, linear_activation_cache = linear_activation_backward_test_case()

    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
    print ("sigmoid:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db) + "\n")

    dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
    print ("relu:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))

    #测试L_model_backward
    print("==============测试L_model_backward==============")
    AL, Y_assess, caches = L_model_backward_test_case()
    grads = L_model_backward(AL, Y_assess, caches)
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dA1 = "+ str(grads["dA1"]))

    print ("dW2 = "+ str(grads["dW2"]))
    print ("db2 = "+ str(grads["db2"]))
    print ("dA2 = "+ str(grads["dA2"]))

    #测试update_parameters
    print("==============测试update_parameters==============")
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))

    print(parameters)


test_case_run()