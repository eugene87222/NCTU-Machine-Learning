#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys

class Matrix:
    def __init__(self, matrix):
        self.m = len(matrix)
        self.n = len(matrix[0])
        self.matrix = []
        for row in matrix:
            self.matrix.append(tuple(row))
    
    def PrintMatrix(self):
        for row in self.matrix:
            print(row)

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
    def Transpose(self):
        result = []
        for i in range(self.n):
            result.append([self.matrix[j][i] for j in range(self.m)])
        return Matrix(result)

    # return matrix1 * matrix2
    def Multiply(self, matrix2):
        result = []
        for row in self.matrix:
            result_row = []
            for col in matrix2.Transpose().matrix:
                result_row.append(sum([i * j for (i, j) in zip(row, col)]))
            result.append(result_row)
        return Matrix(result)

    # return matrix1 + matrix2 * n
    def Add(self, matrix2, n):
        result = []
        for i in range(self.m):
            result.append([a + b * n for (a, b) in zip(self.matrix[i], matrix2.matrix[i])])
        return Matrix(result)
    
    # return matrix^-1
    def Inverse(self):
        # LUA^-1 = I
        L, U = self.LUDecomposition()
        # Ly = I
        y = SolveLy_B(L)
        # Ux = y
        x = SolveUx_y(U, y)
        return x

# return base^exp
def Power(base, exp):
    result = 1
    while exp:
        if exp & 1:
            result *= base
        exp >>= 1
        base *= base
    return result

# return row1 - row2 * n
def RowSubtract(row1, row2, n):
    return [i - n * j for (i, j) in zip(row1, row2)]

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
def IdentityMatrix(n):
    I = []
    for i in range(n):
        I.append([0] * n)
        I[i][i] = 1
    return Matrix(I)

# return total error
def CalculateError(error_matrix):
    result = sum([error[0] * error[0] for error in error_matrix.matrix])
    return result

def Formula(x, coef):
    result = np.zeros(x.shape)
    for i in range(len(coef)):
        result += coef[i][0] * np.power(x, len(coef) - i - 1)
    return result

def PlotResult(coef, l_x, r_x, data_points, type):
    x = np.arange(l_x, r_x, 0.1)
    y = Formula(x, coef)
    plt.subplot(2, 1, type)
    plt.plot(x, y)
    data = np.array(data_points)
    plt.scatter(data[:, 0], data[:, 1], c='r', s=15, edgecolors='k')
    plt.xlim(-6, 6)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ./HW1_class.py <testfile>')
    else:
        # load data points =====================================================
        data_points = []
        with open(f'{sys.argv[1]}', 'r', encoding='utf-8') as file:
            for line in file:
                temp = line.split(',')
                data_points.append((float(temp[0]), float(temp[1])))
        # ======================================================================

        case_num = int(input('Input # of cases: '))
        print('-------------------')

        for _ in range(case_num):
            print(f'Case #{_ + 1}:')

            n = int(input('Input # of polynomial base: '))
            l = float(input('Input lambda: '))
            print()
            
            _A = []
            _b = []
            # generate A, b matrix for Ax = b ==================================
            for point in data_points:
                x_coor = point[0]
                y_coor = point[1]
                row = []
                for exp in range(n - 1, -1, -1):
                    row.append(Power(x_coor, exp))
                _A.append(row)
                _b.append([y_coor])
            # ==================================================================
            
            A = Matrix(_A)
            b = Matrix(_b)
            ATA = A.Transpose().Multiply(A)
            I = IdentityMatrix(ATA.m)
            ATA_lI = ATA.Add(I, l) # for LSE, A^TA + lambda * I
            ATb = A.Transpose().Multiply(b)

            # LSE ==============================================================
            coef_LSE = ATA_lI.Inverse().Multiply(ATb) # (A^TA)^-1)A^Tb
            error_matrix = A.Multiply(coef_LSE).Add(b, -1) # Ax - b
            error = CalculateError(error_matrix)
            # ==================================================================
            
            # print LSE result =================================================
            print('LSE:\nFitting line: ', end='')
            for i in range(coef_LSE.m):
                print(coef_LSE.matrix[i][0], end='')
                if i != coef_LSE.m - 1:
                    print(f'X^{coef_LSE.m - i - 1} + ', end='')
            print(f'\nTotal error: {error}\n')
            # ==================================================================

            # Newton's method ==================================================
            coef_Newton = ATA.Inverse().Multiply(ATb) # (A^TA)^-1)A^Tb
            error_matrix = A.Multiply(coef_Newton).Add(b, -1) # Ax - b
            error = CalculateError(error_matrix)
            # ==================================================================

            # print Newton's method result =====================================
            print('Newton\'s Method:\nFitting line: ', end='')
            for i in range(coef_Newton.m):
                print(coef_Newton.matrix[i][0], end='')
                if i != coef_Newton.m - 1:
                    print(f'X^{coef_Newton.m - i - 1} + ', end='')
            print(f'\nTotal error: {error}\n')
            # ==================================================================

            PlotResult(coef_LSE.matrix, -6, 7, data_points, 1)
            PlotResult(coef_Newton.matrix, -6, 7, data_points, 2)
            plt.show()