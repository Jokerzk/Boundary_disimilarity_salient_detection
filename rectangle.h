#include "BoundaryDissimilarityMap.h"

int getThreshold(cv::Mat img, int width, int height);//get threshold by otsu's method

cv::Rect getrectangular(cv::Mat img,int width, int height);//get rectangular from binary map

cv::Mat CombinationandPostProcessing(cv::Mat salientMap, int width, int height);//get binary map from M's distance

cv::Rect getSalientMap(cv::Mat img, int width, int height);//get the final rectangular

int get_optimize_rect(cv::Mat image, cv::Point2i &tk_pt, cv::Size &tk_sz);