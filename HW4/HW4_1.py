import matplotlib.pyplot as plt
import numpy as np
import sys, math

def UnivariateGaussianDataGenerator(m, std):
    return m + std * (sum(np.random.uniform(0, 1, 12)) - 6)

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

    # return matrix1 + matrix2
    def Add(self, matrix2):
        result = []
        for i in range(self.m):
            result.append([a + b for (a, b) in zip(self.matrix[i], matrix2.matrix[i])])
        return Matrix(result)

    # return matrix1 - matrix2
    def Minus(self, matrix2):
        result = []
        for i in range(self.m):
            result.append([a - b for (a, b) in zip(self.matrix[i], matrix2.matrix[i])])
        return Matrix(result)
    
    # return matrix^-1
    def Inverse(self):
        L, U = self.LUDecomposition()
        y = SolveLy_B(L)
        x = SolveUx_y(U, y)
        return x

    # return det(matrix)
    # def Det(self):
    #     if m == n:
    #         return 0
    #     else:
    #         return 0

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

def Determinant(matrix):
    result = 0
    result += matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2])
    result -= matrix[1][0] * (matrix[0][1] * matrix[2][2] - matrix[2][1] * matrix[0][2])
    result += matrix[2][0] * (matrix[0][1] * matrix[1][2] - matrix[1][1] * matrix[0][2])
    return result

def GenerateDataPointSet(n, mx, vx, my, vy, label):
    data_x = []
    data_y = []
    X = []
    y = []
    for i in range(n):
        xi = UnivariateGaussianDataGenerator(mx, math.sqrt(vx))
        yi = UnivariateGaussianDataGenerator(my, math.sqrt(vy))
        data_x.append(xi)
        data_y.append(yi)
        X.append([xi, yi, 1])
        y.append([label])
    return data_x, data_y, X, y

def Sigmoid(Xw):
    result = []
    for i in range(0, Xw.m):
        row = []
        for j in range(0, Xw.n):
            if Xw.matrix[i][j] <= -709:
                row.append(1 / (1 + math.exp(709)))
            else:
                row.append(1 / (1 + math.exp(-1 * Xw.matrix[i][j])))
        result.append(row)
    return Matrix(result)

def Difference(vector1, vector2):
    good = True
    for (i, j) in zip(vector1, vector2):
        if abs(i - j) > (abs(j) * 0.075):
            good = False
            break
    return good

def MatrixD(Xw):
    D = []
    for i in range(0, Xw.m):
        row = []
        for j in range(0, Xw.n):
            if i == j:
                if Xw.matrix[i][j] <= -709:
                    row.append(math.exp(709) / ((1 + math.exp(709)) ** 2))
                else:
                    row.append(math.exp(-1 * Xw.matrix[i][j]) / ((1 + math.exp(-1 * Xw.matrix[i][j])) ** 2))
            else:
                row.append(0)
        D.append(row)
    return Matrix(D)

learning_rate = 0.05

def GradientDescent(X, y):
    w_pre = Matrix([[0.0], [0.0], [0.0]])
    w = Matrix([[0.0], [0.0], [0.0]])
    XT = X.T()
    while True:
        Xw = X.Mul(w_pre)
        gradient = XT.Mul(y.Minus(Sigmoid(Xw)))
        w = w_pre.Add(gradient.Mul(learning_rate))
        if Difference(w.T().matrix[0], w_pre.T().matrix[0]):
            break
        w_pre = w
    return w

def NewtonsMethod(X, y):
    w_pre = Matrix([[0.0], [0.0], [0.0]])
    w = Matrix([[0.0], [0.0], [0.0]])
    n = X.m
    XT = X.T()
    while True:
        Xw = X.Mul(w_pre).Mul(I(n))
        D = MatrixD(Xw)
        Hessian = X.T().Mul(D.Mul(X))
        gradient = XT.Mul(y.Minus(Sigmoid(Xw)))
        
        if Determinant(Hessian.matrix) == 0:
            w = w_pre.Add(gradient.Mul(learning_rate))
        else:
            w = w_pre.Add(Hessian.Inverse().Mul(gradient))
        if Difference(w.T().matrix[0], w_pre.T().matrix[0]):
            break
        w_pre = w
    return w

def SubplotResult(X, w, y, class1_x, class1_y, class2_x, class2_y, title, separate, subplot_idx):
    if X == None:
        plt.subplot(subplot_idx)
        plt.title(title)
        plt.scatter(class1_x, class1_y, c='r')
        plt.scatter(class2_x, class2_y, c='b')
    else:
        confusion_matrix = [[0, 0], [0, 0]]
        predict = Sigmoid(X.Mul(w))
        predict_class1_x = []
        predict_class1_y = []
        predict_class2_x = []
        predict_class2_y = []
        for i in range(0, predict.m):
            if y.matrix[i][0] == 0:
                if predict.matrix[i][0] < 0.5:
                    predict_class1_x.append(X.matrix[i][0])
                    predict_class1_y.append(X.matrix[i][1])
                    confusion_matrix[0][0] += 1
                else:
                    predict_class2_x.append(X.matrix[i][0])
                    predict_class2_y.append(X.matrix[i][1])
                    confusion_matrix[0][1] += 1
            if y.matrix[i][0] == 1:
                if predict.matrix[i][0] < 0.5:
                    predict_class1_x.append(X.matrix[i][0])
                    predict_class1_y.append(X.matrix[i][1])
                    confusion_matrix[1][0] += 1
                else:
                    predict_class2_x.append(X.matrix[i][0])
                    predict_class2_y.append(X.matrix[i][1])
                    confusion_matrix[1][1] += 1
        print(f'{title}:\n')
        print('w:')
        w.PrintMatrix()
        print('\nConfusion Matrix:')
        print('\t\tPredict cluster 1 Predict cluster 2')
        print(f'Is cluster 1\t\t{confusion_matrix[0][0]}\t\t{confusion_matrix[0][1]}')
        print(f'Is cluster 2\t\t{confusion_matrix[1][0]}\t\t{confusion_matrix[1][1]}')
        sens = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        spec = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
        print(f'\nSensitivity (Successfully predict cluster 1): {sens}')
        print(f'Specificity (Successfully predict cluster 2): {spec}')
        if separate:
            print('\n----------------------------------------')
        plt.subplot(subplot_idx)
        plt.title(title)
        plt.scatter(predict_class1_x, predict_class1_y, c='r')
        plt.scatter(predict_class2_x, predict_class2_y, c='b')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(f'{sys.argv[1]}.txt', 'r', encoding='utf-8') as file:
            input_data = file.readlines()
        N = int(input_data[0])
        mx1 = float(input_data[1])
        vx1 = float(input_data[2])
        my1 = float(input_data[3])
        vy1 = float(input_data[4])
        mx2 = float(input_data[5])
        vx2 = float(input_data[6])
        my2 = float(input_data[7])
        vy2 = float(input_data[8])
    else:
        N = int(input('N: '))
        mx1 = int(input('mean for x1: '))
        vx1 = float(input('variance for x1: '))
        my1 = float(input('mean for y1: '))
        vy1 = float(input('variance for y1: '))
        mx2 = float(input('mean for x2: '))
        vx2 = float(input('variance for x2: '))
        my2 = float(input('mean for y2: '))
        vy2 = float(input('variance for y2: '))

    class1_x, class1_y, X, y = GenerateDataPointSet(N, mx1, vx1, my1, vy1, 0)
    class2_x, class2_y, tempX, tempy = GenerateDataPointSet(N, mx2, vx2, my2, vy2, 1)
    X += tempX
    X = Matrix(X)
    y += tempy
    y = Matrix(y)

    SubplotResult(None, None, None, class1_x, class1_y, class2_x, class2_y, 'Ground truth', None, 131)
    gradient_w = GradientDescent(X, y)
    SubplotResult(X, gradient_w, y, class1_x, class1_y, class2_x, class2_y, 'Gradient descent', True, 132)
    Newton_w = NewtonsMethod(X, y)
    SubplotResult(X, Newton_w, y, class1_x, class1_y, class2_x, class2_y, 'Newton\'s method', False, 133)
    plt.tight_layout()
    plt.show()