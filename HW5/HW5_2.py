import csv
import sys
import time
import numpy as np
from svmutil import *

kernel = {
    'linear': 0, 
    'polynomial': 1, 
    'RBF': 2, 
}

def openCSV(filename):
    with open(filename, 'r') as file:
        content = list(csv.reader(file))
        content = np.array(content)
    return content

def train(Y, X, kernel):
    return svm_train(Y, X, f'-t {kernel} -q')

def compareAccuracy(Y, X, opt, optimal_cv_acc, optimal_opt):
    print(opt)
    cv_acc = svm_train(Y, X, opt)
    if cv_acc > optimal_cv_acc:
        return cv_acc, opt
    else:
        return optimal_cv_acc, optimal_opt

def gridSearch(X, Y):
    optimal_opt = ''
    optimal_cv_acc = 0
    costs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    cnt = 0
    for k in kernel:
        for cost in costs:
            if k == 'linear':
                opt = f'-t {kernel[k]} -c {cost} -v 3 -q'
                cnt += 1
                print(cnt)
                optimal_cv_acc, optimal_opt = compareAccuracy(Y, X, opt, optimal_cv_acc, optimal_opt)
            elif k == 'polynomial':
                for gamma in gammas:
                    for degree in range(2, 5):
                        for coef0 in range(0, 3):
                            opt = f'-t {kernel[k]} -c {cost} -g {gamma} -d {degree} -r {coef0} -v 3 -q'
                            cnt += 1
                            print(cnt)
                            optimal_cv_acc, optimal_opt = compareAccuracy(Y, X, opt, optimal_cv_acc, optimal_opt)
            elif k == 'RBF':
                for gamma in gammas:
                    opt = f'-t {kernel[k]} -c {cost} -g {gamma} -v 3 -q'
                    cnt += 1
                    print(cnt)
                    optimal_cv_acc, optimal_opt = compareAccuracy(Y, X, opt, optimal_cv_acc, optimal_opt)
    print(f'Total combinations: {cnt}')
    print(f'Optimal cross validation accuracy: {optimal_cv_acc}')
    print(f'Optimal option: {optimal_opt}')

def linearKernel(X1, X2):
    kernel = X1 @ X2.T
    return kernel
    
def RBFKernel(X1, X2, gamma):
    dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
    kernel = np.exp((-1 * gamma * dist))
    return kernel

if __name__ == '__main__':
    START = time.time()

    if len(sys.argv) < 2:
        print('Please choose a part to run')
        print('Part 1. compare three types of kernel')
        print('Part 2. grid search')
        print('Part 3. user-defined kernel')
        print('Usage: python HW5_2.py {number of part}')
        exit()

    X_train = openCSV('X_train.csv').astype(np.float64)
    Y_train = list(openCSV('Y_train.csv').astype(np.int32).ravel())
    X_test = openCSV('X_test.csv').astype(np.float64)
    Y_test = list(openCSV('Y_test.csv').astype(np.int32).ravel())

    if sys.argv[1] == '1':
        # Part 1. compare three types of kernel
        # for k in kernel:
        #     print(f'Kernel: {k}')
        #     m = train(Y_train, X_train, kernel[k])
        #     res = svm_predict(Y_test, X_test, m)
        m =  svm_train(Y_train, X_train, f'-t 1 -d 2 -q')
        res = svm_predict(Y_test, X_test, m)
    elif sys.argv[1] == '2':
        # Part 2. grid search
        gridSearch(X_train, Y_train)

        opt = '-t 2 -c 10 -g 0.01 -q'
        m = svm_train(Y_train, X_train, opt)
        res = svm_predict(Y_test, X_test, m)
    elif sys.argv[1] == '3':
        # Part 3. user-defined kernel
        linear_kernel = linearKernel(X_train, X_train)
        RBF_kernel = RBFKernel(X_train, X_train, 1 / 784)
        linear_kernel_s = linearKernel(X_train, X_test).T
        RBF_kernel_s = RBFKernel(X_train, X_test, 1 / 784).T

        X_kernel = np.hstack((np.arange(1, 5001).reshape((-1, 1)), linear_kernel + RBF_kernel))
        X_kernel_s = np.hstack((np.arange(1, 2501).reshape((-1, 1)), linear_kernel_s + RBF_kernel_s))
    
        opt = '-t 4 -q'
        m = svm_train(Y_train, X_kernel, opt)
        svm_predict(Y_test, X_kernel_s, m)
    else:
        print('Wrong part number')
    
    print(f'Spend {time.time() - START} second(s)')