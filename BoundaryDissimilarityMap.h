#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>  
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define k 3  

struct Tuple{
	float attrL;
	float attra;
	float attrb;
};
struct Group{
	vector<Tuple> clusters[k];
	Tuple means[k];
	float covinvert[k][3][3];
};

void Normalize(float* input, float* output, const int width, const int height, const int normrange);

cv::Mat getBoundaryDissimilarityMap(cv::Mat Src, int boundarysize);

/*functions for K_means*/
vector<Tuple>* KMeans(cv::Mat imgLab, int width, int height, int boundarysize, vector<Tuple> clusters[k], Tuple means[k]);

vector<Tuple> getSeeds(cv::Mat imgLab, int width, int height, vector<Tuple> Seeds);

float getDistLab(Tuple t1, Tuple t2);

int clusterOfTuple(Tuple means[], Tuple tuple);

void getCovmatrixInvert(int width, int height, vector<Tuple> clusters[k],float covinvert[k][3][3]);

Tuple getMeans(vector<Tuple> cluster);