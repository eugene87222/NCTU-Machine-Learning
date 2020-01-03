import numpy as np
from PIL import Image
import time, re, os, sys
import scipy.spatial
import matplotlib.pyplot as plt

POINT_NUM = 10000
LENGTH = 100
CLUSTER_NUM = 2
colormap = ['black', 'darkorange', 'cornflowerblue', 'silver', 
	'gold', 'green', 'navy', 'magenta', 'yellow', 'red', 'peru', 'pink', 
	'dodgerblue', 'greenyellow', 'cyan']
FOLDER = ''

def openImage(filename):
	img = Image.open(filename)
	img = np.array(img)
	return img

def computeKernel(image):
	gamma_s = 1 / (LENGTH * LENGTH)
	gamma_c = 1 / (100 * 100)
	coor = []
	for i in range(LENGTH):
		for j in range(LENGTH):
			coor.append([i, j])
	coor = np.array(coor)
	s = np.exp(-gamma_s * scipy.spatial.distance.cdist(coor, coor, 'sqeuclidean'))
	c = np.exp(-gamma_c * scipy.spatial.distance.cdist(image, image, 'sqeuclidean'))
	return s * c

def init(eigen, method):
	cluster = np.random.randint(0, CLUSTER_NUM, POINT_NUM)
	if method == 'random':
		high = eigen.max(axis=0)
		low = eigen.min(axis=0)
		interval = high - low
		centroids =	np.random.rand(CLUSTER_NUM, CLUSTER_NUM)
		for i in range(CLUSTER_NUM):
			centroids[:, i] *= interval[i]
			centroids[:, i] += low[i]
	elif method == 'kmeans++':
		centroids = [eigen[np.random.choice(range(POINT_NUM)), :]]
		for i in range(CLUSTER_NUM - 1):
			dist = scipy.spatial.distance.cdist(eigen, centroids, 'euclidean').min(axis=1)
			prob = dist / np.sum(dist)
			centroids.append(eigen[np.random.choice(range(POINT_NUM), p=prob)])
		centroids = np.array(centroids)
	return centroids, np.array(cluster)

def clustering(eigen, centroids):
	cluster = np.zeros(POINT_NUM, dtype=int)
	for i in range(POINT_NUM):
		distance = np.zeros(CLUSTER_NUM, dtype=np.float32)
		for j in range(CLUSTER_NUM):
			distance[j] = np.sum(np.absolute(eigen[i] - centroids[j]))
		cluster[i] = np.argmin(distance)
	return cluster

def computeError(cluster, pre_cluster):
	error = np.sum(np.absolute(cluster - pre_cluster))
	return error

def updateCentroids(eigen, centroids, cluster):
	centroids = np.zeros(centroids.shape, dtype=np.float64)
	cnt = np.zeros(CLUSTER_NUM, dtype=np.int32)
	for i in range(POINT_NUM):
		centroids[cluster[i]] += eigen[i]
		cnt[cluster[i]] += 1
	for i in range(CLUSTER_NUM):
		if cnt[i] == 0:
			cnt[i] = 1
		centroids[i] /= cnt[i]
	return centroids

def draw(image, imagename, cluster, iteration, cut):
	title = f'Spectral Clustering Iteration-{iteration}'
	plt.clf()
	plt.subplot(121, aspect='equal')
	plt.imshow(image)
	plt.subplot(122, aspect='equal')
	cluster_x = []
	cluster_y = []
	for j in range(CLUSTER_NUM):
		cluster_x.append([])
		cluster_y.append([])
	for i in range(POINT_NUM):
		cluster_x[cluster[i]].append((i // LENGTH) / (LENGTH - 1))
		cluster_y[cluster[i]].append((i % LENGTH) / (LENGTH - 1))
	for j in range(CLUSTER_NUM):
		plt.scatter(cluster_y[j], cluster_x[j], s=2, c=[colormap[j]])
	plt.xlim(0, 1)
	plt.ylim(1, 0)
	plt.suptitle(title)
	plt.tight_layout()
	plt.savefig(f'./spectral-clustering/{FOLDER}/{imagename}_{CLUSTER_NUM}-cluster_{cut}_{iteration}.png')

def drawEigenSpace(eigen, cluster, cut):
	plt.clf()
	title = "Eigen-Space"
	cluster_x = []
	cluster_y = []
	for j in range(CLUSTER_NUM):
		cluster_x.append([])
		cluster_y.append([])
	for i in range(POINT_NUM):
		cluster_x[cluster[i]].append(eigen[i][0])
		cluster_y[cluster[i]].append(eigen[i][1])
	for i in range(CLUSTER_NUM):
		plt.scatter(cluster_x[i], cluster_y[i], s=2, c=[colormap[i]])
	plt.title(title)
	plt.tight_layout()
	plt.savefig(f'./spectral-clustering/{FOLDER}/{imagename}_{CLUSTER_NUM}-cluster_{cut}_eigenspace.png')

def kMeans(image, imagename, eigen, cut, method):
	iteration = 0
	centroids, cluster = init(eigen, method)
	draw(image, imagename, cluster, iteration, cut)
	while 1:
		iteration += 1
		pre_cluster = cluster
		cluster = clustering(eigen, centroids)
		centroids = updateCentroids(eigen, centroids, cluster)
		draw(image, imagename, cluster, iteration, cut)
		error = computeError(cluster, pre_cluster)
		print(f'Iter: {iteration}: {error}')
		if error == 0:
			break
	if CLUSTER_NUM == 2:
		drawEigenSpace(eigen, cluster, cut)

if __name__ == "__main__":
	start = time.time()
	if len(sys.argv) < 5:
		print(f'python3 {sys.argv[0]} <k-cluster> <imagename> {{normalized|ratio}} {{kmeans++|random}}')
	else:
		CLUSTER_NUM = int(sys.argv[1])
		imagename = sys.argv[2]
		cut = sys.argv[3]
		method = sys.argv[4]
		FOLDER = f'{imagename}_{CLUSTER_NUM}-cluster_{cut}_{method}'
		if not os.path.exists(f'spectral-clustering/{FOLDER}'):
			os.mkdir(f'spectral-clustering/{FOLDER}')
		image = openImage(f'{imagename}.png')
		kernel = computeKernel(image.reshape(-1, 3))
		D = np.sum(kernel, axis=1)
		if cut == 'normalized':
			D_sqr = np.diag(np.power(D, -0.5))
			L = np.identity(POINT_NUM) - D_sqr @ kernel @ D_sqr
			eigen_val, eigen_vec = np.linalg.eig(L)
			idx = np.argsort(eigen_val)
			eigen_vec = eigen_vec[:, idx]
			U = eigen_vec[:, 1:CLUSTER_NUM+1].real
			T = np.zeros(U.shape, dtype=np.float64)
			for i in range(POINT_NUM):
				T[i] = U[i] / np.sqrt(np.sum(U[i] ** 2))
			kMeans(image, imagename, T, cut, method)
		elif cut == 'ratio':
			L = D - kernel
			eigen_val, eigen_vec = np.linalg.eig(L)
			idx = np.argsort(eigen_val)
			eigen_vec = eigen_vec[:, idx]
			U = eigen_vec[:, 1:CLUSTER_NUM+1].real
			kMeans(image, imagename, U, cut, method)
		else:
			print('unknown cut')
	print(time.time() - start)
