import numpy as np
from matplotlib import pyplot as plt


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gp import GPR_predictor

def f(X):
    return X * np.sin(X)


def main():
    ### Generate dataset
    x_train = np.array([1., 3., 5., 6., 7., 8.])
    y_train = f(x_train)

    x_new = np.linspace(0, 10, 1000)


    ### plot ovserbed data and original function    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x_train, y_train, 'r.', markersize=16)
    plt.plot(x_new, f(x_new), 'k')
    ### save
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16)
    plt.ylim(-8, 12)
    plt.legend(['$y = x*\sin(x)$', 'Observed values'], loc='upper left', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.savefig("plots/data.png")
    plt.close()


    ### Predict
    predictor = GPR_predictor()
    predictor.train(x_train, y_train)
    pred_mu, pred_sigma = predictor.predict(x_new)


    ### plot ovserbed data and original function
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x_train, y_train , 'r.', markersize=16)
    plt.plot(x_new, f(x_new), 'k')
    ### plot predicted mean
    plt.plot(x_new, pred_mu, 'b')
    ### plot predicted std
    p_val = 1.96
    plt.fill_between(x_new.squeeze(), (pred_mu - p_val * pred_sigma).squeeze(), (pred_mu + p_val * pred_sigma).squeeze())
    ### save
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16)
    plt.ylim(-8, 12)
    plt.legend(['$y = x*\sin(x)$', 'Observed values', 'Predicted mean', '95% confidence interval'], loc='upper left', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.savefig("plots/result.png")
    plt.close()




