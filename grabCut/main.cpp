#include <vector>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <time.h>
#include "interact.h"
#include "grabCut.h"
#include "graph.h"
using namespace std;
using namespace cv;

void test();

int main() {
	Mat img = imread("test2.jpg");
	GrabCut gc;
	gc.init(img);
	gc.run();
	Mat result;
	Mat matte = gc.getMatte();
	img.copyTo(result, matte);
	imshow("result", result);
	gc.interact();
	cout << "interact\n";


	Mat matte2 = gc.getMatte();
	Mat result2;
	img.copyTo(result2, matte2);
	imshow("result", result2);
	waitKey();
	// test();
	return 0;
}

pair<double, double> precisionAndRecall(const Mat& gt, const Mat& pred) {
	Point p;
	int precision = 0, recall = 0, trueSample = 0;
	int rows = gt.rows, cols = gt.cols;
	for (p.y = 0; p.y < rows; p.y++) {
		for (p.x = 0; p.x < cols; p.x++) {
			if (pred.at<uchar>(p) == gt.at<uchar>(p)) {
				precision++;
			}
			if (gt.at<uchar>(p) == 255) {
				trueSample++;
				if (pred.at<uchar>(p) == 255)
					recall++;
			}
		}
	}
	return make_pair(double(precision) / (cols * rows), double(recall) / trueSample);
}

void test() {
	const string gtPath = "test/gt/GT";
	const string inputPath = "test/input/GT";
	const string trimapPath = "test/trimap/GT";
	long exec = 0;
	Mat input, trimap, gt;
	double precision = 0.0, recall = 0.0;
	GrabCut gc;
	for (int i = 0; i < 27; i++) {
		cout << "Image No." << i + 1 << endl;
		string name = (i < 9) ? string("0") + char('1' + i) : to_string(i + 1);
		input = imread(inputPath + name + ".png");
		trimap = imread(trimapPath + name + ".png", IMREAD_GRAYSCALE);
		gt = imread(gtPath + name + ".png", IMREAD_GRAYSCALE);
		gc.init(input, true);
		gc.setInitMatte(trimap);
		
		long beg = time(NULL);
		gc.run();
		long end = time(NULL);
		exec += end - beg;

		Mat result = gc.getMatte();
		imshow("res", result);
		waitKey();
		auto precAndReca = precisionAndRecall(gt, result);
		precision += precAndReca.first;
		recall += precAndReca.second;
	}
	cout << "Percision: " << precision / 27 << endl;
	cout << "Recall: " << recall / 27 << endl;
	cout << "Execution Time: " << double(exec) / 27 << "s\n";
}