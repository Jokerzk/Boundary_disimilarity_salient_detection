#include "rectangle.h"

int getThreshold(cv::Mat img, int width, int height)
{
	int size = width*height;

	cv::MatND outputhist;
	int hisSize[1] = { 256 };
	float range[2] = { 0.0, 255.0 };
	const float *ranges; ranges = &range[0];
	calcHist(&img, 1, 0, Mat(), outputhist, 1, hisSize, &ranges);
	double sum = 0;
	for (int i = 0; i < 256; i++)
	{
		sum = sum + i * outputhist.at<float>(i);
	}
	int threshold = 0;
	float sumvaluew = 0.0, sumvalue = 0.0, maximum = 0.0, wF, p12, diff, between;
	for (int i = 0; i < 256; i++)
	{
		sumvalue = sumvalue + outputhist.at<float>(i);
		sumvaluew = sumvaluew + i * outputhist.at<float>(i);
		wF = size - sumvalue;
		p12 = wF * sumvalue;
		if (p12 == 0){ p12 = 1; }
		diff = sumvaluew * wF - (sum - sumvaluew) * sumvalue;
		between = (float)diff * diff / p12;
		if (between >= maximum){
			threshold = i;
			maximum = between;
		}
	}
	return threshold;
}

cv::Rect getrectangular(cv::Mat img,int width, int height)
{
	
	IplImage pImg = IplImage(img);
	CvMemStorage * storage = cvCreateMemStorage(0);
	CvSeq * contour = 0, *contmax = 0,*contmaxold = 0;
	cvFindContours(&pImg, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	double area = 0, maxArea = 0, maxAreaold = 0, distf = 0, dists = 0;
	Rect maxrectf, maxrects;
	int count = 0;
	for (; contour; contour = contour->h_next)
	{
		area = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
		if (area > maxArea)
		{
			contmaxold = contmax;
			contmax = contour;
			maxArea = area;
		}
		if (contmaxold == NULL && area >= maxAreaold && area < maxArea)
		{
			contmaxold = contour;
			maxAreaold = area;
		}
		count++;
	}
	maxrectf = cvBoundingRect(contmax, 0);//此处寻找距离中心最大的连通域
	if (count > 1)
	{
		maxrects = cvBoundingRect(contmaxold, 0);
		if (maxrects.width < width / 4 && maxrects.height < height / 4)
		{
			return maxrectf;
		}
		distf = ((maxrectf.x + maxrectf.width / 2) - width / 2)*((maxrectf.x + maxrectf.width / 2) - width / 2) + ((maxrectf.y + maxrectf.height / 2) - height / 2)*((maxrectf.y + maxrectf.height / 2) - height / 2);
		dists = ((maxrects.x + maxrects.width / 2) - width / 2)*((maxrects.x + maxrects.width / 2) - width / 2) + ((maxrects.y + maxrects.height / 2) - height / 2)*((maxrects.y + maxrects.height / 2) - height / 2);
		if (distf> dists)//距离中心距离近者返回
		{
			return maxrects;
		}
	}
	return maxrectf;
}

cv::Mat CombinationandPostProcessing(cv::Mat salientMap, int width, int height)
{
	int threshold = getThreshold(salientMap, width, height);
	cv::Mat binaryMap = cv::Mat(salientMap.rows, salientMap.cols, CV_8UC1);
	for (int i = 0; i < width*height; i++)
	{	
		if (salientMap.data[i] >= threshold)
			binaryMap.data[i] =255;
		else
			binaryMap.data[i] = 0;
	}
	return binaryMap;
}

cv::Rect getSalientMap(cv::Mat img, int width, int height)
{
	Mat elementerode = getStructuringElement(MORPH_CROSS, Size(6, 6));
	Mat elementdilate = getStructuringElement(MORPH_RECT, Size(3, 3));
	cv::Mat salientMap = getBoundaryDissimilarityMap(img, 4);
	cv::erode(salientMap, salientMap, elementerode);//一次腐蚀
	cv::imshow("salient", salientMap);  
	waitKey();

	cv::Mat binaryMap = CombinationandPostProcessing(salientMap, width, height);//根据距离图进行阈值求解和二值化分割
	Mat out;
	
	cv::dilate(binaryMap, out, elementerode);
	//cv::erode(out, out, elementerode);
	cv::dilate(out, out, elementdilate);//一次腐蚀+两次膨胀
	cv::imshow("binary", binaryMap);
	waitKey();
	//cv::erode(out, out, elementerode);
	
	cv::Rect salrect;
	salrect = getrectangular(out,width,height);


	return salrect;
}

int get_optimize_rect(cv::Mat image, cv::Point2i &tk_pt, cv::Size &tk_sz)
{
	cv::Rect init_rect;
	init_rect.x = tk_pt.x - tk_sz.width / 2 < 0 ? 0 : tk_pt.x - tk_sz.width / 2;
	init_rect.y = tk_pt.y - tk_sz.height / 2 < 0 ? 0 : tk_pt.y - tk_sz.height / 2;
	init_rect.width = tk_sz.width < image.cols - init_rect.x ? tk_sz.width : image.cols - init_rect.x;
	init_rect.height = tk_sz.height < image.rows - init_rect.y ? tk_sz.height : image.rows - init_rect.y;

	cv::Mat roimat = image(init_rect);
	cv::Size fixed_size(120, 120);
	cv::resize(roimat, roimat, fixed_size);
	cv::Rect optimize_rect = getSalientMap(roimat, 120, 120);
	//长细比钝化
	if (optimize_rect.width < optimize_rect.height / 3)
	{
		optimize_rect.width = optimize_rect.width * 0.4*(optimize_rect.height / optimize_rect.width);
		optimize_rect.x = optimize_rect.x - optimize_rect.width * 0.2*(optimize_rect.height / optimize_rect.width -0.5);

		optimize_rect.x = optimize_rect.x < 0 ? 0 : optimize_rect.x;
		optimize_rect.width = optimize_rect.x + optimize_rect.width < 120 ? optimize_rect.width : 120 - optimize_rect.x;
	}
	if (optimize_rect.height < optimize_rect.width / 3)
	{
		optimize_rect.height = optimize_rect.height * 0.4*(optimize_rect.width / optimize_rect.height);
		optimize_rect.y = optimize_rect.y - optimize_rect.height * 0.2*(optimize_rect.width / optimize_rect.height -0.5);

		optimize_rect.y = optimize_rect.y < 0 ? 0 : optimize_rect.y;
		optimize_rect.height = optimize_rect.y + optimize_rect.height < 120 ? optimize_rect.height : 120 - optimize_rect.y;
	}
	float ratiox = init_rect.width / 120.0;
	float ratioy = init_rect.height / 120.0;
	optimize_rect.x = optimize_rect.x * ratiox + 0.5;
	optimize_rect.y = optimize_rect.y * ratioy + 0.5;
	optimize_rect.width = optimize_rect.width * ratiox + 0.5;
	optimize_rect.height = optimize_rect.height * ratioy + 0.5;

	//cv::resize(roimat, roimat, tk_sz);
	cv::rectangle(image, optimize_rect, CV_RGB(255, 0, 0), 2);
	cv::imshow("roi", image);
	cv::waitKey(0);

	if (optimize_rect.width > init_rect.width / 4 || optimize_rect.height > init_rect.height / 4)
	{
		tk_pt.x = init_rect.x + optimize_rect.x + optimize_rect.width / 2;
		tk_pt.y = init_rect.y + optimize_rect.y + optimize_rect.height / 2;
		tk_sz.width = optimize_rect.width;
		tk_sz.height = optimize_rect.height;
	}
	return 0;
}

int main()
{
	string str;
	printf("Plz input an image \n");
	getline(cin,str);
	cv::Mat img = imread(str);
	int width = img.cols;
	int height = img.rows;
	cv::Point2i ini_pt;
	ini_pt.x = width * 0.5;
	ini_pt.y = height* 0.5;
	cv::Size ini_sz;
	ini_sz.width = width;
	ini_sz.height = height;

	get_optimize_rect(img, ini_pt, ini_sz);

	return 0;
}