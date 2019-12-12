import numpy as np
from numba import jit
from numba.errors import NumbaWarning
import warnings, time
warnings.simplefilter('ignore', category=NumbaWarning)

n_images = 60000
n_rows = 28
n_cols = 28
image_size = n_rows * n_cols

@jit
def GetData():	
    images = np.fromfile('./data/train-images.idx3-ubyte', dtype='ubyte')
    X = (images[4 * 4:].astype('float64').reshape(n_images, -1) / 128).astype('int32')
    labels = np.fromfile('./data/train-labels.idx1-ubyte',dtype='ubyte')
    labels = labels[2 * 4:].astype('int')
    return X, labels

@jit
def Init():
    LAMBDA = np.full((10), 0.1, dtype=np.float64)
    P = np.random.rand(10, image_size).astype(np.float64)
    P_prev = np.zeros((10, image_size), dtype=np.float64)
    return LAMBDA, P, P_prev

@jit
def E(X, LAMBDA, P):
    Z = np.zeros((n_images, 10), dtype=np.float64)
    for i in range(n_images):
        marginal = 0
        for j in range(10):
            p = LAMBDA[j]
            for k in range(image_size):
                p *= (P[j, k] ** X[i, k])
                p *= ((1 - P[j, k]) ** (1 - X[i, k]))
            Z[i, j] = p
            marginal += p
        if marginal == 0:
            continue
        Z[i, :] /= marginal
    return Z

@jit
def M(X, Z, LAMBDA, P):
    for i in range(10):
        n = np.sum(Z[:, i])
        LAMBDA[i] = n / 60000
        if n == 0:
            n = 1
        for j in range(image_size):
            P[i, j] = np.dot(X[:, j], Z[:, i]) / n
    return LAMBDA, P

@jit
def Check(LAMBDA, condition):
    if 0 in LAMBDA:
        return 0
    else:
        return condition + 1

@jit
def AssignLabel(X, labels, LAMBDA, P):
    counting = np.zeros((10, 10), dtype=np.int32)
    mapping = np.zeros((10), dtype=np.int32)
    for i in range(n_images):
        p = np.zeros((10), dtype=np.float64)
        for j in range(10):
            p_ = LAMBDA[j]
            for k in range(image_size):
                p_ *= (P[j, k] ** X[i, k])
                p_ *= ((1 - P[j, k]) ** (1 - X[i, k]))
            p[j] = p_
        counting[labels[i], np.argmax(p)] += 1
    for i in range(10):
        idx = np.argmax(counting)
        label = idx // 10 
        class_idx = idx % 10 
        mapping[label] = class_idx
        counting[label, :] = -1
        counting[:, class_idx] = -1
    return mapping

@jit
def PrintImagination(P, mapping, labeled):
    for i in range(10):
        if labeled:
            print('labeled ', end='')
        print('class: ' + str(i))
        class_idx = mapping[i]
        for r in range(n_rows):
            for c in range(n_cols):
                char = '1' if P[class_idx, r * n_cols + c] >= 0.5 else '0'
                print(char, end=' ')
            print()
        print()

@jit
def PrintResult(X, labels, LAMBDA, P, mapping, iteration):
    PrintImagination(P, mapping, True)
    mapping_inverse = np.zeros((10), dtype=np.int32)
    for i in range(10):
        mapping_inverse[i] = np.where(mapping == i)[0][0]
    confusion_matrix = np.zeros((10, 2, 2), dtype=np.int)
    for i in range(n_images):
        p = np.zeros((10), dtype=np.float64)
        for j in range(10):
            p_ = LAMBDA[j]
            for k in range(image_size):
                p_ *= (P[j, k] ** X[i, k])
                p_ *= ((1 - P[j, k]) ** (1 - X[i, k]))
            p[j] = p_
        prediction = mapping_inverse[np.argmax(p)]
        for j in range(10):
            if labels[i] == j:
                if prediction == j:
                    confusion_matrix[j][0][0] += 1
                else:
                    confusion_matrix[j][0][1] += 1
            else:
                if prediction == j:
                    confusion_matrix[j][1][0] += 1
                else:
                    confusion_matrix[j][1][1] += 1
    for i in range(10):
        print('---------------------------------------------------------------\n')
        print('Confusion Matrix ' + str(i) + ':')
        print('\t\tPredict number ' + str(i) + ' Predict not number ' + str(i))
        print('Is number ' + str(i) + '\t\t' + str(confusion_matrix[i][0][0]) + '\t\t' + str(confusion_matrix[i][0][1]))
        print('Isn\'t number ' + str(i) + '\t\t' + str(confusion_matrix[i][1][0]) + '\t\t' + str(confusion_matrix[i][1][1]))
        sens = confusion_matrix[i][0][0] / (confusion_matrix[i][0][0] + confusion_matrix[i][0][1])
        spec = confusion_matrix[i][1][1] / (confusion_matrix[i][1][0] + confusion_matrix[i][1][1])
        print('\nSensitivity (Successfully predict number ' + str(i) + ')\t: ' + str(sens))
        print('Specificity (Successfully predict not number ' + str(i) + ')\t: ' + str(spec) + '\n')
    
    error = n_images - np.sum(confusion_matrix[:, 0, 0])
    print('Total iteration to converge: ' + str(iteration))
    print('Total error rate: ' + str(error / 60000))
    
if __name__ == '__main__':
    START = time.time()
    X, labels = GetData()
    LAMBDA, P, P_prev = Init()
    iteration = 0
    max_iteration = 200
    condition = 0
    mapping = np.array([i for i in range(10)], dtype=np.int32)
    while iteration < max_iteration:
        iteration += 1
        Z = E(X, LAMBDA, P)
        LAMBDA, P = M(X, Z, LAMBDA, P)
        condition = Check(LAMBDA, condition)
        if condition == 0:
            LAMBDA, P, _ = Init()
        delta = sum(sum(abs(P - P_prev)))
        PrintImagination(P, mapping, False)
        print(f'No. of Iteration: {iteration}, Difference: {delta}\n')
        print('------------------------------------------------------------')
        if delta < 1e-5 and condition >= 10 and np.sum(LAMBDA) >= 0.95:
            break
        P_prev = P
        print()
    print('------------------------------------------------------------\n')
    mapping = AssignLabel(X, labels, LAMBDA, P)
    PrintResult(X, labels, LAMBDA, P, mapping, iteration)
    # print(time.time() - START)