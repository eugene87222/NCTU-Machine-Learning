import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det, cholesky
from scipy.optimize import minimize

def LoadData(filename):
    X = []
    Y = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            x, y = line.split(' ')
            X.append(float(x))
            Y.append(float(y))
    X = np.array(X, dtype=np.float64).reshape(-1, 1)
    Y = np.array(Y, dtype=np.float64).reshape(-1, 1)
    return X, Y

def RationalQuadraticKernel(X1, X2, sigma, alpha, length_scale):
    dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
    kernel = (sigma ** 2) * ((1 + dist / (2 * alpha * (length_scale ** 2))) ** (-1 * alpha))
    return kernel

def NegativeLogLikelihood(theta, X, Y, beta):
    theta = theta.ravel()
    kernel = RationalQuadraticKernel(X, X, theta[0], theta[1], theta[2])
    kernel += np.identity(len(X), dtype=np.float64) * (1 / beta)
    nll = np.sum(np.log(np.diagonal(cholesky(kernel))))
    nll += 0.5 * Y.T @ inv(kernel) @ Y
    nll += 0.5 * len(X) * np.log(2 * np.pi)
    return nll

def GaussianProcess(X, Y, X_s, beta, sigma, alpha, length_scale):
    kernel = RationalQuadraticKernel(X, X, sigma, alpha, length_scale) 
    C = kernel + np.identity(len(X), dtype=np.float64) * (1 / beta)
    C_inv = inv(C)
    kernel_s = RationalQuadraticKernel(X, X_s, sigma, alpha, length_scale)
    kernel_ss = RationalQuadraticKernel(X_s, X_s, sigma, alpha, length_scale) 

    mu_s = kernel_s.T @ C_inv @ Y
    var_s = kernel_ss + np.identity(len(X_s), dtype=np.float64) * (1 / beta)
    var_s -= kernel_s.T @ C_inv @ kernel_s
    
    plt.plot(X_s, mu_s, color='b')
    plt.scatter(X, Y, color='k', s=10)
    
    interval = 1.96 * np.sqrt(np.diag(var_s))
    X_s = X_s.ravel()
    mu_s = mu_s.ravel()

    plt.plot(X_s, mu_s + interval, color='r')
    plt.plot(X_s, mu_s - interval, color='r')
    plt.fill_between(X_s, mu_s + interval, mu_s - interval, color='r', alpha=0.1)

    plt.title(f'sigma: {sigma:.5f}, alpha: {alpha:.5f}, length scale: {length_scale:.5f}')
    plt.xlim(-60, 60)
    plt.show()

if __name__ == '__main__':
    '''
    X: 34 x 1
    Y: 34 x 1
    kernel(X, X): 34 x 34
    '''

    X, Y = LoadData('input.data')
    X_s = np.linspace(-60.0, 60.0, 1000).reshape(-1, 1)
    beta = 5
    sigma = 1
    alpha = 1
    length_scale = 1
    GaussianProcess(X, Y, X_s, beta, sigma, alpha, length_scale)
    
    sigma = 1
    alpha = 1
    length_scale = 1
    opt = minimize(NegativeLogLikelihood, [sigma, alpha, length_scale], 
                    bounds=((1e-8, 1e6), (1e-8, 1e6), (1e-8, 1e6)), 
                    args=(X, Y, beta))
    sigma_opt = opt.x[0]
    alpha_opt = opt.x[1]
    length_scale_opt = opt.x[2]
    GaussianProcess(X, Y, X_s, beta, sigma_opt, alpha_opt, length_scale_opt)