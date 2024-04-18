import copy
import math

import numpy as np
from pandas import read_csv
import csv
import matplotlib.pyplot as plt

def load_data_pandas():
    d = read_csv('train_copy.csv')
    data_array = d.values
    return data_array

def transform_data(array):
    #we want to change the Y/N to 0/1
    sub_array = array[:, 19:20]
    dimensions = sub_array.shape
    m,n = dimensions
    for i in range(m):
        if sub_array[i] == 'Y':
            sub_array[i] = 0
        else:
            sub_array[i] = 1
    return array.astype(float)

def scatter_plot(X_train, y_train):
    m,n = X_train.shape
    for i in range(m):
        for j in range(n):
            plt.scatter(X_train[i,j], y_train[i])
    plt.show()
def load_data_training_set(array):
    X_train = array[:,2:23]
    y_train = array[:,23]
    return X_train, y_train

def load_data_testing_training_set():
    d = read_csv('test_copy.csv')
    test_data = d.values
    test_data = transform_data(test_data)
    new_test_data = test_data[:,2:24]
    return new_test_data

def normalize_test_data(test_data,mu,sigma):
    X_test_data_norm = (test_data - mu)/sigma
    return X_test_data_norm

def z_score_norm_data(X_train):
    mu = np.mean(X_train, axis=0)   #mu and sigma will have shape (n,)
    sigma = np.std(X_train, axis=0)
    X_train_norm = (X_train - mu)/sigma
    print("Here is mu ", mu)
    print("Here is sigma ", sigma)
    print("Here is X Train Normalized first 5: ", X_train_norm[:5])
    return (X_train_norm, mu, sigma)

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2*m)
    return cost

def compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i],w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in) #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        #calculate the gradient and update params
        dj_dw, dj_db = gradient_function(X, y, w, b)

        #Update Params using w, b, alpha, and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        #Save cost J at each iteration
        if i < 100000:
            cost = cost_function(X, y, w, b)
            J_history.append(cost_function(X,y,w,b))
            if i % math.ceil(num_iters / 10) == 0:
                print("Iteration ", i, "Cost is ", cost)
    return w, b, J_history

def predict_house_price(X, w_final, b_final):
    m = X.shape[0]
    for i in range(m):
        f_wb = np.dot(X[i], w_final) + b_final
        print("Prediction of house price is ", f_wb)

if __name__ == "__main__":
   data_array = load_data_pandas()
   data_array = transform_data(data_array)
   X_train, y_train = load_data_training_set(data_array)
   print("X_train is ", X_train[:5])
   print("y_train is ", y_train[:5])
   print("NOW WE NORMALIZE THE DATA ! ----------------")
   X_train_norm, mu, sigma = z_score_norm_data(X_train)
   w_init = np.zeros([21])
   b_init = 0
   alpha = 0.01
   iterations = 5000
   print("Cost is ", compute_cost(X_train_norm,y_train, w_init,b_init))
   tmp_dj_dw, tmp_dj_db = compute_gradient(X_train_norm, y_train, w_init, b_init)
   print ("Lets find our final w and b !")
   w_final, b_final, J_History = gradient_descent(X_train_norm,y_train,w_init,b_init,compute_cost, compute_gradient, alpha, iterations)
   print ( "w_final is ", w_final, "Our b is ", + b_final)
   print("So now we can predict using our w and b from our training set, first lets normalize our test data !")
   X_test = load_data_testing_training_set()
   X_test_norm = normalize_test_data(X_test,mu,sigma)
   predict_house_price(X_test_norm, w_final, b_final)








