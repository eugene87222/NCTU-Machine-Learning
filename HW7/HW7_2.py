import sys, os
import numpy as np
import seaborn as sns
import scipy.spatial.distance
import matplotlib.pyplot as plt

def plotResult(Y, labels, idx, interval, method, perplexity):
    plt.clf()
    scatter = plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.legend(*scatter.legend_elements(), loc='lower left', title='Digit')
    plt.title(f'{method}, perplexity: {perplexity}, iteration: {idx}')
    plt.tight_layout()
    if interval:
        plt.savefig(f'./{method}_{perplexity}/{idx // interval}.png')
    else:
        plt.savefig(f'./{method}_{perplexity}/{idx}.png')

def Hbeta(D, beta):
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X, tol, perplexity):
    print('Computing pairwise distances...')
    (n, d) = X.shape
    D = scipy.spatial.distance.cdist(X, X, 'sqeuclidean')
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print(f'Computing P-values for point {i} of {n}...')
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    print(f'Mean value of sigma: {np.mean(np.sqrt(1 / beta))}')
    return P

def pca(X, dims):
    print('Preprocessing the data using PCA...')
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, axis=0), (n, 1))
    (eigen_val, eigen_vec) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, eigen_vec[:, 0:dims])
    return Y

def sne(X, dims, init_dims, perplexity, labels, method, interval):
    X = pca(X, init_dims).real
    (n, d) = X.shape
    Y = np.random.randn(n, dims)
    dY = np.zeros((n, dims))
    iY = np.zeros((n, dims))
    gains = np.ones((n, dims))
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4                   # early exaggeration
    P = np.maximum(P, 1e-12)

    for itr in range(max_iter):
        # Compute pairwise affinities
        if method == 'tsne':
            num = 1 / (1 + scipy.spatial.distance.cdist(Y, Y, 'sqeuclidean'))
        else:
            num = np.exp(-1 * scipy.spatial.distance.cdist(Y, Y, 'sqeuclidean'))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            if method == 'tsne':
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (dims, 1)).T * (Y[i, :] - Y), axis=0)
            else:
                dY[i, :] = np.sum(np.tile(PQ[:, i], (dims, 1)).T * (Y[i, :] - Y), axis=0)

        # Perform the update
        if itr < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        if itr % interval == 0:
            plotResult(Y, labels, itr, interval, method, perplexity)

        # Compute current value of cost function
        if (itr + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print(f'Iteration {itr + 1}: error is {C}')

        # Stop lying about P-values
        if itr == 100:
            P = P / 4

    return Y, P, Q

def plotHighDLowD(P, Q, method, perplexity):
    pal = sns.light_palette('blue', as_cmap=True)
    plt.clf()
    plt.title('High-D Similarity')
    plt.imshow(P, cmap=pal)
    plt.savefig(f'./{method}_{perplexity}/High-D.png')

    plt.clf()
    plt.title('Low-D Similarity')
    plt.imshow(Q, cmap=pal)
    plt.savefig(f'./{method}_{perplexity}/Low-D.png')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'Usage: python3 {sys.argv[0]} {{tsne|ssne}} <perplexity> <interval>')
    else:
        X = np.loadtxt('mnist2500_X.txt')
        labels = np.loadtxt('mnist2500_labels.txt')
        method = sys.argv[1]
        perplexity = float(sys.argv[2])
        interval = int(sys.argv[3])
        if not os.path.exists(f'{method}_{perplexity}'):
            os.mkdir(f'{method}_{perplexity}')
        if method == 'tsne' or method == 'ssne':
            print(f'Running {method} on 2500 MNIST digits...')
            Y, P, Q = sne(X, 2, 50, perplexity, labels, method, interval)
            plotResult(Y, labels, 'final', None, method, perplexity)
            plotHighDLowD(P, Q, method, perplexity)
        else:
            print('Unknown method')