import sys, math, binascii
import numpy as np

def LoadImage(file):
    image_file = open(f'./data/{file}', 'rb')
    image_file.read(4) # magic number
    return image_file

def LoadLabel(file):
    label_file = open(f'./data/{file}', 'rb')
    label_file.read(4) # magic number
    return label_file

def PrintResult(prob, answer):
    print('Posterior (in log scale):')
    for i in range(prob.shape[0]):
        print(f'{i}: {prob[i]}')
    pred = np.argmin(prob)
    print(f'Prediction: {pred}, Ans: {answer}\n')
    return 0 if answer == pred else 1

def DrawImagination(image, image_row, image_col, mode):
    print('Imagination of numbers in Baysian classifier\n')
    if mode == 0: # discrete
        for i in range(10):
            print(f'{i}:')
            for j in range(image_row):
                for k in range(image_col):
                    white = sum(image[i][j * image_row + k][:17])
                    black = sum(image[i][j * image_row + k][17:])
                    print(f'{1 if black > white else 0} ', end='')
                print()
            print()
    elif mode == 1: # continuous
        for i in range(10):
            print(f'{i}:')
            for j in range(image_row):
                for k in range(image_col):
                    print(f'{1 if image[i][j * image_row + k] > 128 else 0} ', end='')
                print()
            print()

def LoadTrainingData():
    train_image_file = LoadImage('train-images.idx3-ubyte')
    train_size = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    image_row = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    image_col = int(binascii.b2a_hex(train_image_file.read(4)), 16)
    train_label_file = LoadLabel('train-labels.idx1-ubyte')
    train_label_file.read(4)
    return train_image_file, train_label_file, train_size, image_row, image_col

def LoadTestingData():
    test_image_file = LoadImage('t10k-images.idx3-ubyte')
    test_size = int(binascii.b2a_hex(test_image_file.read(4)), 16)
    test_image_file.read(4)
    test_image_file.read(4)
    test_label_file = LoadLabel('t10k-labels.idx1-ubyte')
    test_label_file.read(4)
    return test_image_file, test_label_file, test_size

def DiscreteMode():
    train_image_file, train_label_file, train_size, image_row, image_col = LoadTrainingData()
    test_image_file, test_label_file, test_size = LoadTestingData()

    image_size = image_row * image_col
    image = np.zeros((10, image_size, 32), dtype=np.int32)
    image_sum = np.zeros((10, image_size), dtype=np.int32)
    prior = np.zeros((10), dtype=np.int32)

    for i in range(train_size):
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)
        prior[label] += 1
        for j in range(image_size):
            grayscale = int(binascii.b2a_hex(train_image_file.read(1)), 16)
            image[label][j][grayscale // 8] += 1
            image_sum[label][j] += 1

    # testing
    error = 0
    for i in range(test_size):
        # print(i, error)
        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)
        prob = np.zeros((10), dtype=np.float)
        test_image = np.zeros((image_size), dtype=np.int32)
        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16)
        for j in range(10):
            # consider prior
            prob[j] += np.log(prior[j] / train_size)
            for k in range(image_size):
                # consider likelihood
                likelihood = image[j][k][test_image[k] // 8]
                if likelihood == 0:
                    likelihood = np.min(image[j][k][np.nonzero(image[j][k])])
                # likelihood = 0.000001 if likelihood == 0 else likelihood
                prob[j] += np.log(likelihood / image_sum[j][k])
        # normalize
        summation = sum(prob)
        prob /= summation
        error += PrintResult(prob, answer)

    DrawImagination(image, image_row, image_col, 0)
    print(f'Error rate: {error / test_size}')

def ContinuousMode():
    train_image_file, train_label_file, train_size, image_row, image_col = LoadTrainingData()
    test_image_file, test_label_file, test_size = LoadTestingData()

    image_size = image_row * image_col    
    prior = np.zeros((10), dtype=np.int32)
    var = np.zeros((10, image_size), dtype=np.float)
    mean = np.zeros((10, image_size), dtype=np.float)
    mean_square = np.zeros((10, image_size), dtype=np.float)

    for i in range(train_size):
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)
        prior[label] += 1
        for j in range(image_size):
            grayscale = int(binascii.b2a_hex(train_image_file.read(1)), 16)
            mean[label][j] += grayscale
            mean_square[label][j] += (grayscale ** 2)
    
    for i in range(10):
        for j in range(image_size):
            mean[i][j] /= prior[i]
            mean_square[i][j] /= prior[i]
            var[i][j] = mean_square[i][j] - (mean[i][j] ** 2)
            var[i][j] = 1000 if var[i][j] == 0 else var[i][j]

    # testing
    error = 0
    for i in range(test_size):
        # print(i, error)
        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)
        prob = np.zeros((10), dtype=np.float)
        test_image = np.zeros((image_size), dtype=np.int32)
        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16)
        for j in range(10):
            # consider prior
            prob[j] += np.log(prior[j] / train_size)
            for k in range(image_size):
                # consider likelihood
                likelihood = -0.5 * (np.log(2 * math.pi * var[j][k]) + ((test_image[k] - mean[j][k]) ** 2) / var[j][k])
                prob[j] += likelihood
        # normalize
        summation = sum(prob)
        prob /= summation
        error += PrintResult(prob, answer)

    DrawImagination(mean, image_row, image_col, 1)
    print(f'Error rate: {error / test_size}')

if __name__ == '__main__':
    mode = input('Discrete(0) or continuous(1): ')
    print(mode)
    if mode == '0':
        try:
            with open('Discrete', 'r', encoding='utf-8') as file:
                for line in file:
                    print(line, end='')
        except:
            DiscreteMode()
    elif mode == '1':
        try:
            with open('Continuous', 'r', encoding='utf-8') as file:
                for line in file:
                    print(line, end='')
        except:
            ContinuousMode()