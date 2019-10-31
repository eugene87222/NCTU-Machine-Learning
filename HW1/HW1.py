#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys

def PrintMatrix(matrix):
    for row in matrix:
        print(row)

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

# return L, U -> matrix = LU (lower-upper decomposition)
def LUDecomposition(matrix):
    size = len(matrix)
    L = []
    for i in range(size):
        L.append([0] * size)
    U = []
    for row in matrix:
        U.append(list(row))
    for i in range(0, size, 1):
        L[i][i] = 1.0
        for j in range(i + 1, size, 1):
            scalar = U[j][i] / U[i][i]
            L[j][i] = scalar
            temp = RowSubtract(U[j], U[i], scalar)
            U[j] = temp    
    L = [tuple(row) for row in L]
    U = [tuple(row) for row in U]
    return L, U

# return matrix^T
def Transpose(matrix):
    m = len(matrix)
    n = len(matrix[0])
    result = []
    for i in range(n):
        result.append(tuple([matrix[j][i] for j in range(m)]))
    return result

# return matrix1 * matrix2
def MatrixMultiplication(matrix1, matrix2):
    result = []
    for row in matrix1:
        result_row = []
        for col in Transpose(matrix2):
            result_row.append(sum([i * j for (i, j) in zip(row, col)]))
        result.append(tuple(result_row))
    return result

# return matrix1 + matrix2 * n
def MatrixAddition(matrix1, matrix2, n):
    result = []
    for i in range(len(matrix1)):
        result.append([i + j * n for (i, j) in zip(matrix1[i], matrix2[i])])
    result = [tuple(row) for row in result]
    return result

# return y -> Ly = B
def SolveLy_B(L):
    size = len(L)
    y = []
    for i in range(size):
        y.append([0] * size)
    for i in range(0, size):
        for j in range(0, size):
            B = 1.0 if i == j else 0.0
            for k in range(0, j):
                B -= (L[j][k] * y[k][i])
            y[j][i] = B
    y = [tuple(row) for row in y]
    return y

# return x -> Ux = y
def SolveUx_y(U, y):
    size = len(U)
    x = []
    for i in range(size):
        x.append([0] * size)
    for i in range(size - 1, -1, -1):
        for j in range(size - 1, -1, -1):
            B = y[j][i]
            for k in range(j + 1, size):
                B -= (U[j][k] * x[k][i])
            x[j][i] = B / U[j][j]
    x = [tuple(row) for row in x]
    return x

# return matrix^-1
def Inverse(matrix):
    L, U = LUDecomposition(matrix)
    y = SolveLy_B(L)
    x = SolveUx_y(U, y)
    return x

# return n x n identity matrix
def IdentityMatrix(n):
    I = []
    for i in range(n):
        I.append([0] * n)
        I[i][i] = 1
    I = [tuple(row) for row in I]
    return I

# return total error
def CalculateError(error_matrix):
    result = sum([error[0] * error[0] for error in error_matrix])
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
        print('Usage: python ./HW1.py <testfile>')
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
            A = []
            b = []
            # generate A, b matrix for Ax = b ==================================
            for point in data_points:
                x_coor = point[0]
                y_coor = point[1]
                row = []
                for exp in range(n - 1, -1, -1):
                    row.append(Power(x_coor, exp))
                A.append(tuple(row))
                b.append(tuple([y_coor]))
            # ==================================================================
            
            ATA = MatrixMultiplication(Transpose(A), A)
            ATA_lI = MatrixAddition(ATA, IdentityMatrix(len(ATA)), l) # for LSE
            ATb = MatrixMultiplication(Transpose(A), b)

            # LSE ==============================================================
            coef_LSE = MatrixMultiplication(Inverse(ATA_lI), ATb)
            error_matrix = MatrixAddition(MatrixMultiplication(A, coef_LSE), b, -1)
            error = CalculateError(error_matrix)
            # ==================================================================
            
            # print LSE result =================================================
            print('LSE:\nFitting line: ', end='')
            for i in range(len(coef_LSE)):
                print(coef_LSE[i][0], end='')
                if i != len(coef_LSE) - 1:
                    print(f'X^{len(coef_LSE) - i - 1} + ', end='')
            print(f'\nTotal error: {error}\n')
            # ==================================================================

            # Newton's method ==================================================
            coef_Newton = MatrixMultiplication(Inverse(ATA), ATb)
            error_matrix = MatrixAddition(MatrixMultiplication(A, coef_Newton), b, -1)
            error = CalculateError(error_matrix)
            # ==================================================================

            # print Newton's method result =====================================
            print('Newton\'s Method:\nFitting line: ', end='')
            for i in range(len(coef_Newton)):
                print(coef_Newton[i][0], end='')
                if i != len(coef_Newton) - 1:
                    print(f'X^{len(coef_Newton) - i - 1} + ', end='')
            print(f'\nTotal error: {error}\n')
            # ==================================================================

            PlotResult(coef_LSE, -6, 7, data_points, 1)
            PlotResult(coef_Newton, -6, 7, data_points, 2)
            plt.show()