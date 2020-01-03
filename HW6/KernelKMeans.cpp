#include <climits>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

int CLUSTER_NUM;
int POINT_NUM = 10000;
int LENGTH = 100;

vector<pair<int, int>> centroid;
vector<int> cluster;
vector<int> pre_cluster;

void loadImage(string imagename, int image[][3]) {
    ifstream infile(imagename);
    for(int i = 0; i < POINT_NUM; i++) {
        infile >> image[i][0] >> image[i][1] >> image[i][2];
    }
    infile.close();
}

void init(string method) {
    int x, y, c;
    centroid = vector<pair<int, int>>(CLUSTER_NUM);
    cluster = vector<int>(POINT_NUM);
    pre_cluster = vector<int>(POINT_NUM, 0);
    if(method == "random") {
        for(int i = 0; i < POINT_NUM; i++) {
            c = rand() % CLUSTER_NUM;
            cluster[i] = c;
        }
    }
    else if(method == "mod") {
        for(int i = 0; i < POINT_NUM; i++) {
            cluster[i] = i % CLUSTER_NUM;
        }
    }
    else if(method == "kmeans++") {
        x = rand() % LENGTH, y = rand() % LENGTH;
        centroid[0] = make_pair(x, y);
        for(int i = 1; i < CLUSTER_NUM; i++) {
            int max_x, max_y, max_distance = 0;
            int dx, dy, distance;
            for(int j = 0; j < LENGTH; j++) {
                for(int k = 0; k < LENGTH; k++) {
                    dx = centroid[i - 1].first - j;
                    dy = centroid[i - 1].second - k;
                    distance = dx * dx + dy * dy;
                    if(distance > max_distance) {
                        max_distance = distance;
                        max_x = j, max_y = k;
                    }
                }
            }
            centroid[i] = make_pair(max_x, max_y);
        }
        for(int i = 0; i < POINT_NUM; i++) {
            float distance[CLUSTER_NUM];
            int dx, dy;
            for(int j = 0; j < CLUSTER_NUM; j++) {
                dx = centroid[j].first - (i / 100);
                dy = centroid[j].second - (i % 100);
                distance[j] = dx * dx + dy* dy;
            }
            int min_idx, min_distance = INT_MAX;
            for(int j = 0; j < CLUSTER_NUM; j++) {
                if(distance[j] < min_distance) {
                    min_idx = j;
                    min_distance = distance[j];
                }
            }
            cluster[i] = min_idx;
        }
    }
}

void computeKernel(int image[][3], float *kernel[]) {
    float i_x, i_y, j_x, j_y;
    float i_r, i_g, i_b, j_r, j_g, j_b;
    float spatial, color;
    float gamma_s = 1.0 / (LENGTH * LENGTH);
    float gamma_c = 1.0 / (255 * 255);
    for(int i = 0; i < POINT_NUM; i++) {
        i_x = (i / LENGTH), i_y = (i % LENGTH);
        i_r = image[i][0], i_g = image[i][1], i_b = image[i][2];
        for(int j = 0; j < POINT_NUM; j++) {
            j_x = (j / LENGTH), j_y = (j % LENGTH);
            j_r = image[j][0], j_g = image[j][1], j_b = image[j][2];
            spatial = -1 * gamma_s * pow(i_x - j_x, 2);
            spatial += -1 * gamma_s * pow(i_y - j_y, 2);
            color = -1 * gamma_c * pow(i_r - j_r, 2);
            color += -1 * gamma_c * pow(i_g - j_g, 2);
            color += -1 * gamma_c * pow(i_b - j_b, 2);
            kernel[i][j] = exp(spatial + color);
        }
    }
}

float second(float *kernel[], int image_idx, int cluster_idx) {
	float result = 0;
	int cluster_size = 0;
    for(int i = 0; i < POINT_NUM; i++) {
        if(cluster[i] == cluster_idx) {
            cluster_size += 1;
			result += kernel[image_idx][i];
        }
    }
	if(cluster_size == 0) {
		cluster_size = 1;
    }
	return -2 * (result / cluster_size);
}

vector<float> third(float *kernel[]) {
    vector<float> C_k(CLUSTER_NUM, 0);
    vector<float> third_k(CLUSTER_NUM, 0);
    for(int i = 0; i < POINT_NUM; i++) {
        C_k[cluster[i]] += 1;
    }
    for(int i = 0; i < CLUSTER_NUM; i++) {
        for(int p = 0; p < POINT_NUM; p++) {
            for(int q = p + 1; q < POINT_NUM; q++) {
                if(cluster[p] == i && cluster[q] == i) {
                    third_k[i] += kernel[p][q];
                }
            }
        }
    }
    for(int i = 0; i < CLUSTER_NUM; i++) {
        if(C_k[i] == 0) {
            C_k[i] = 1;
        }
        third_k[i] /= (C_k[i] * C_k[i]);
    }
    return third_k;
}

void clustering(float *kernel[]) {
    vector<float> third_k = third(kernel);
    for(int i = 0; i < POINT_NUM; i++) {
        float distance[CLUSTER_NUM];
        for(int j = 0; j < CLUSTER_NUM; j++) {
            distance[j] = second(kernel, i, j) + third_k[j];
        }
        int min_idx, min_distance = INT_MAX;
        for(int j = 0; j < CLUSTER_NUM; j++) {
            if(distance[j] < min_distance) {
                min_idx = j;
                min_distance = distance[j];
            }
        }
        cluster[i] = min_idx;
    }
}

int computeError() {
    int error = 0;
    for(int i = 0; i < POINT_NUM; i++) {
        error += abs(cluster[i] - pre_cluster[i]);
    }
    return error;
}

void outputCluster(int iter) {
    ofstream outfile;
    string filename = "./kernel-k-means/cluster/" + to_string(CLUSTER_NUM);
    filename += "-cluster_iter-" + to_string(iter);
    outfile.open(filename);
    for(int i = 0; i < POINT_NUM; i++) {
        outfile << i << " " << cluster[i] << "\n";
    }
    outfile.close();
}

void kernelKMeans(float *kernel[], string method) {
    init(method);
    int iter = 0, error;
    error = computeError();
    cout << "Iter: " << iter << ": " << error << endl;
    outputCluster(iter);
    while(error != 0) {
        iter += 1;
        pre_cluster = cluster;
        clustering(kernel);
        outputCluster(iter);
        error = computeError();
        cout << "Iter: " << iter << ": " << error << endl;
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    if(argc < 4) {
        cout << argv[0] << "<k-cluster> <imagename> {kmeans++|mod|random}\n";
    }
    else {
        string imagename, method;
        CLUSTER_NUM = atoi(argv[1]);
        imagename = string(argv[2]);
        method = string(argv[3]);
        int image[POINT_NUM][3];
        float *kernel[POINT_NUM];
        for(int i = 0; i < POINT_NUM; i++) {
            kernel[i] = new float[POINT_NUM];
        }
        loadImage(imagename, image);
        computeKernel(image, kernel);
        kernelKMeans(kernel, method);
    }
    return 0;
}