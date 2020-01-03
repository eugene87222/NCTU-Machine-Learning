import numpy as np
from PIL import Image
import time, re, os, sys
import matplotlib.pyplot as plt

FOLDER = ''
POINT_NUM = 10000
LENGTH = 100
CLUSTER_NUM = 2
colormap = ['black', 'darkorange', 'cornflowerblue', 'silver', 
	'gold', 'green', 'navy', 'magenta', 'yellow', 'red', 'peru', 'pink', 
	'dodgerblue', 'greenyellow', 'cyan']

def openImage(filename):
	img = Image.open(filename)
	img = np.array(img)
	return img

def getClusterFile():
	cluster_file = []
	for item in os.listdir('./kernel-k-means/cluster'):
		if re.findall(f'{CLUSTER_NUM}-cluster', item):
			cluster_file.append(item)
	return cluster_file

def drawGIF(imagename):
	cluster_file = getClusterFile()
	cluster_file.sort(key=lambda x: int(re.findall(r'\d-cluster_iter-(\d+)', x)[0]))
	image = openImage(f'{imagename}.png')
	for i in range(len(cluster_file)):
		title = f'Kernel KMeans Iteration-{i}'
		plt.clf()
		plt.subplot(121, aspect='equal')
		plt.imshow(image)
		plt.subplot(122, aspect='equal')
		cluster_x = []
		cluster_y = []
		cluster_color = []
		for j in range(CLUSTER_NUM):
			cluster_x.append([])
			cluster_y.append([])
		with open(f'./kernel-k-means/cluster/{cluster_file[i]}', 'r') as file:
			for line in file:
				if line.strip():
					idx, label = line.split(' ')
					cluster_x[int(label)].append((int(idx) // LENGTH) / 99)
					cluster_y[int(label)].append((int(idx) % LENGTH) / 99)
				else:
					break
		for j in range(CLUSTER_NUM):
			plt.scatter(cluster_y[j], cluster_x[j], s=2, c=[colormap[j]])
		plt.xlim(0, 1)
		plt.ylim(1, 0)
		plt.suptitle(title)
		plt.tight_layout()
		plt.savefig(f'./kernel-k-means/{FOLDER}/{imagename}_{CLUSTER_NUM}-cluster_{i}.png')
		print(cluster_file[i])

if __name__ == "__main__":
	start = time.time()
	if len(sys.argv) < 4:
		print(f'python3 {sys.argv[0]} <k-cluster> <imagename> {{kmeans++|mod|random}}')
	else:
		CLUSTER_NUM = int(sys.argv[1])
		imagename = sys.argv[2]
		method = sys.argv[3]
		FOLDER = f'{imagename}_{CLUSTER_NUM}-cluster_{method}'
		if not os.path.exists(f'kernel-k-means/{FOLDER}'):
			os.mkdir(f'kernel-k-means/{FOLDER}')
		drawGIF(imagename)
	print(time.time() - start)