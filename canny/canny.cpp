#include "canny.h"
#include <vector>
using namespace std;
using namespace cv;

#define HORIZONTAL 0
#define VERTICAL 1
#define DIAGONAL 2
#define BACK_DIAGONAL 3


inline static int dir(const double degree) {
	if (abs(degree) <= 22.5 || abs(degree - 180) <= 22.5)
		return VERTICAL;
	else if (abs(degree - 90) <= 22.5 || abs(degree + 90) <= 22.5)
		return HORIZONTAL;
	else if (abs(degree + 45) <= 22.5 || abs(degree - 135) <= 22.5)
		return DIAGONAL;
	else if (abs(degree - 45) <= 22.5 || abs(degree + 135) <= 22.5)
		return BACK_DIAGONAL;
	return -1;
}

inline static bool isInRange(const int r, const int c, const int rows, const int cols) {
	return r > -1 && r < rows && c > -1 && c < cols;
}

inline bool isMax(const double curr, const double cmp_op1, const double cmp_op2) {
	return curr >= cmp_op1 && curr >= cmp_op2;
}

void calMagnitude(const Mat& src, Mat& mag, int kSize, normMode mode) {
	int rows = src.rows, cols = src.cols;
	Mat dx, dy, tmp_mag;
	mag = Mat::zeros(Size(cols, rows), CV_64FC1);
	Sobel(src, dx, CV_64FC1, 1, 0, kSize, 1, 0, BORDER_DEFAULT);
	Sobel(src, dy, CV_64FC1, 0, 1, kSize, 1, 0, BORDER_DEFAULT);
	if (mode == normMode::L1) {
		tmp_mag = dx + dy;
	}
	else if (mode == normMode::L2) {
		magnitude(dx, dy, tmp_mag);
	}
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			double x = dx.at<double>(r, c);
			double y = dy.at<double>(r, c);
			double ang = atan2(y, x) / CV_PI * 180;
			double cur_mag = tmp_mag.at<double>(r, c);
			// Not Max Surpress
			switch (dir(ang)) {
			case VERTICAL: {
				
				double left = isInRange(r, c - 1, rows, cols) 
								? tmp_mag.at<double>(r, c - 1) : 0;
				double right = isInRange(r, c + 1, rows, cols)
								? tmp_mag.at<double>(r, c + 1) : 0;
				if (isMax(cur_mag, left, right))
					mag.at<double>(r, c) = cur_mag;
				break;
			}
			case HORIZONTAL: {
				double above = isInRange(r - 1, c, rows, cols)
								? tmp_mag.at<double>(r - 1, c) : 0;
				double below = isInRange(r + 1, c , rows, cols)
								? tmp_mag.at<double>(r + 1, c) : 0;
				if (isMax(cur_mag, above, below))
					mag.at<double>(r, c) = cur_mag;
				break;
			}
			case DIAGONAL: {
				double left_above = isInRange(r - 1, c - 1, rows, cols)
								? tmp_mag.at<double>(r - 1, c - 1) : 0;
				double right_below = isInRange(r + 1, c + 1, rows, cols)
								? tmp_mag.at<double>(r + 1, c + 1) : 0;
				if (isMax(cur_mag, left_above, right_below))
					mag.at<double>(r, c) = cur_mag;
				break;
			}
			case BACK_DIAGONAL: {
				double right_above = isInRange(r - 1, c + 1, rows, cols)
								? tmp_mag.at<double>(r - 1, c + 1) : 0;
				double left_below = isInRange(r + 1, c - 1, rows, cols)
								? tmp_mag.at<double>(r + 1, c - 1) : 0;
				if (isMax(cur_mag, right_above, left_below))
					mag.at<double>(r, c) = cur_mag;
				break;
			}
			}
		}
	}
}

void traceEdge(const Mat& mag, Mat& edges, double TL, int r, int c) {
	int rows = mag.rows, cols = mag.cols;
	if (edges.at<uchar>(r, c) == 0) {
		edges.at<uchar>(r, c) = 255;
		for (int i = -1; i < 2; i++) {
			for (int j = -1; j < 2; j++) {
				if (isInRange(r + i, c + j, rows, cols) && mag.at<double>(r + i, c + j) >= TL) {
					traceEdge(mag, edges, TL, r + i, c + j);
				}
			}
		}
	}
}

void canny(const Mat& src, Mat& dst, double TL, double TH, int kSize, normMode mode) {
	int rows = src.rows, cols = src.cols;
	Mat mag = Mat::zeros(rows, cols, CV_64FC1);
	GaussianBlur(src, dst, Size(3, 3), 0.8);
	dst = src;
	calMagnitude(dst, mag, kSize, mode);
	dst = Mat::zeros(rows, cols, CV_8UC1);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			double cur_mag = mag.at<double>(r, c);
			if (cur_mag >= TH) {
				traceEdge(mag, dst, TL, r, c);
			}
		}
	}
}