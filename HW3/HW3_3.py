#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys, math, time

def UnivariateGaussianDataGenerator(m, std):
    return m + std * (sum(np.random.uniform(0, 1, 12)) - 6)

def PolynomialBasisLinearModelDataGenerator(n, std_err, w):
    x = np.random.uniform(-1, 1)
    y = sum([w[i] * (x ** i) for i in range(n)]) + UnivariateGaussianDataGenerator(0, std_err)
    return float(x), float(y)

class Matrix:
    def __init__(self, matrix):
        self.m = len(matrix)
        self.n = len(matrix[0])
        self.matrix = [tuple(row) for row in matrix]
    
    def PrintMatrix(self):
        for row in self.matrix:
            print(','.join([f'{val:>15.10f}' for val in row]))
    
    # return L, U -> matrix = LU (lower-upper decomposition)
    def LUDecomposition(self):
        size = self.m
        L = []
        for i in range(size):
            L.append([0] * size)
        U = []
        for row in self.matrix:
            U.append(list(row))
        for i in range(0, size, 1):
            L[i][i] = 1.0
            for j in range(i + 1, size, 1):
                scalar = U[j][i] / U[i][i]
                L[j][i] = scalar
                U[j] = [val_1 - scalar * val_2 for (val_1, val_2) in zip(U[j], U[i])]
        return Matrix(L), Matrix(U)
    
    # return matrix^T
    def T(self):
        result = []
        for i in range(self.n):
            result.append([self.matrix[j][i] for j in range(self.m)])
        return Matrix(result)

    # return matrix1 * sth
    def Mul(self, sth):
        result = []
        if type(sth) == int or type(sth) == float:
            for row in self.matrix:
                result.append([val * sth for val in row])
        else:
            for row in self.matrix:
                result_row = []
                for col in sth.T().matrix:
                    result_row.append(sum([i * j for (i, j) in zip(row, col)]))
                result.append(result_row)
        return Matrix(result)

    # return matrix1 + matrix2 * n
    def Add(self, matrix2):
        result = []
        for i in range(self.m):
            result.append([a + b for (a, b) in zip(self.matrix[i], matrix2.matrix[i])])
        return Matrix(result)
    
    # return matrix^-1
    def Inverse(self):
        L, U = self.LUDecomposition()
        y = SolveLy_B(L)
        x = SolveUx_y(U, y)
        return x

# return y -> Ly = B
def SolveLy_B(L):
    size = L.m
    y = []
    for i in range(size):
        y.append([0] * size)
    for i in range(0, size):
        for j in range(0, size):
            B = 1.0 if i == j else 0.0
            for k in range(0, j):
                B -= (L.matrix[j][k] * y[k][i])
            y[j][i] = B
    return Matrix(y)

# return x -> Ux = y
def SolveUx_y(U, y):
    size = U.m
    x = []
    for i in range(size):
        x.append([0] * size)
    for i in range(size - 1, -1, -1):
        for j in range(size - 1, -1, -1):
            B = y.matrix[j][i]
            for k in range(j + 1, size):
                B -= (U.matrix[j][k] * x[k][i])
            x[j][i] = B / U.matrix[j][j]
    return Matrix(x)

# return n x n identity matrix
def I(n):
    I = []
    for i in range(n):
        I.append([0] * n)
        I[i][i] = 1
    return Matrix(I)

def SubplotResult(idx, title, x, y, m, a, lambda_inverse, err_var, ground_truth):
    plt.subplot(idx)
    plt.title(title)
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)
    function = np.poly1d(np.flip(m))
    x_curve = np.linspace(-2.0, 2.0, 30)
    y_curve = function(x_curve)
    plt.plot(x_curve, y_curve, 'k')
    if ground_truth:
        plt.plot(x_curve, y_curve + err_var, 'r')
        plt.plot(x_curve, y_curve - err_var, 'r')
    else:
        plt.scatter(x, y, s=10)
        y_curve_plus_var = []
        y_curve_minus_var = []
        for i in range(0, 30):    
            X = Matrix([[x_curve[i] ** j for j in range(0, n)]])
            distance = (1 / a) + X.Mul(Matrix(lambda_inverse)).Mul(X.T()).matrix[0][0]
            y_curve_plus_var.append(y_curve[i] + distance)
            y_curve_minus_var.append(y_curve[i] - distance)
        plt.plot(x_curve, y_curve_plus_var, 'r')
        plt.plot(x_curve, y_curve_minus_var, 'r')

if __name__ == '__main__':
    ### input data #############################################################
    b = int(input('Input b: '))
    n = int(input('Input n: '))
    a_err = float(input('Input a: '))
    w = [float(val.strip()) for val in input('Input w: ').split(',')]
    ############################################################################
    START = time.time()
    
    m = Matrix([[0] for i in range(0, n)]) # mean of posterior
    m_pre = Matrix([[0] for i in range(0, n)]) # mean of posterior of previous interation
    x = [] # x coordinates
    y = [] # y coordinates
    num = 0 # number of (z, y) points
    
    var_predict = 0 # variance of predictive distribution
    var_predict_pre = 0 # variance of predictive distribution of previous iteration

    while True:
        new_x, new_y = PolynomialBasisLinearModelDataGenerator(n, math.sqrt(a_err), w)
        num += 1
        x.append(new_x)
        y.append(new_y)
        print(f'Add data point ({new_x:>.5f}, {new_y:>.5f}):\n')

        var = 0
        for i in range(len(x)):
            temp = Matrix([[x[i] ** j for j in range(0, n)]])
            var += (y[i] - temp.Mul(m_pre).matrix[0][0]) ** 2
        var /= num
        a = 1 / (0.00000001 if var == 0 else var)
        
        X = Matrix([[new_x ** i for i in range(0, n)]])
        new_y = Matrix([[new_y]])
        if num == 1:
            LAMBDA = X.T().Mul(X).Mul(a).Add(I(n).Mul(b)) # Λ = aX^TX+bI
            m = LAMBDA.Inverse().Mul(X.T()).Mul(new_y).Mul(a) # μ = aΛ^-1X^TY
        else:
            C = X.T().Mul(X).Mul(a).Add(LAMBDA) # C = aX^TX+Λ
            m = C.Inverse().Mul(X.T().Mul(new_y).Mul(a).Add(LAMBDA.Mul(m_pre))) # μ = C^-1(aX^TY+Λμ)
            LAMBDA = C
        
        m_predict = X.Mul(m).matrix[0][0]
        var_predict = X.Mul(LAMBDA.Inverse()).Mul(X.T()).matrix[0][0] + (1 / a)
        
        print('Posterior mean:')
        m.PrintMatrix()
        
        print('\nPosterior variance:')
        LAMBDA.Inverse().PrintMatrix()
        
        print(f'\nPredictive distribution ~ N({m_predict:>.5f}, {var_predict:>.5f})')
        print('--------------------------------------------------')
        
        if abs(var_predict_pre - var_predict) < 1e-4 and num >= 1000:
            break
        
        m_pre = m
        var_predict_pre = var_predict
        
        if num == 10:
            m_10 = m.matrix.copy()
            x_10 = x.copy()
            y_10 = y.copy()
            a_10 = a
            lambda_inverse_10 = LAMBDA.Inverse().matrix.copy()
        elif num == 50:
            m_50 = m.matrix.copy()
            x_50 = x.copy()
            y_50 = y.copy()
            a_50 = a
            lambda_inverse_50 = LAMBDA.Inverse().matrix.copy()
    
    print(time.time() - START)
    SubplotResult(221, 'Ground truth',     None, None, w,                       None, None,                    a_err, True)
    SubplotResult(222, 'Predict result',   x,    y,    np.reshape(m.matrix, n), a,    LAMBDA.Inverse().matrix, None,  False)
    SubplotResult(223, 'After 10 incomes', x_10, y_10, np.reshape(m_10, n),     a_50, lambda_inverse_10,       None,  False)
    SubplotResult(224, 'After 50 incomes', x_50, y_50, np.reshape(m_50, n),     a_50, lambda_inverse_50,       None,  False)
    plt.tight_layout()
    plt.show()