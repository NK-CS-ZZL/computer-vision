#pragma once
#pragma warning(disable: 26439)
#pragma warning(disable: 26495)
#pragma warning(disable: 26812)
#pragma warning(disable: 26451)
#pragma warning(disable: 6201)
#pragma warning(disable: 6294)
#pragma warning(disable: 6269)
#include "opencv2/opencv.hpp"
#include <vector>

enum class cutFlag { INTERNAL, EXTERNAL };

using std::vector;
using cv::Mat;
using cv::Rect;
using cv::Size;

void imgCut(const Mat& src, Mat& dst, Rect range, cutFlag flag = cutFlag::INTERNAL);

Mat magnitude(const Mat& mat);

void dft(const Mat& src, Mat& dst);

void idft(const Mat& src, Mat& dst);

Mat logAndNormal(const Mat& mat);

void diagonalExchange(Mat& mat);

void highPassFliter(const Mat& src, Mat& dst, const double percent = 0.125, vector<Mat> * mag = nullptr);

void lowPassFliter(const Mat& src, Mat& dst, const double percent = 0.125, vector<Mat>* mag = nullptr);

void _hybrid(const Mat& mHp, const Mat& mLp, Mat& res, const double percent = 0.125, vector<Mat>* mag = nullptr);

Mat combine(const vector<Mat>& imgs, int rowNum, int colNum, Size newSize);

void formatDisplay(const vector<Mat>& srcs, const Mat& res, const vector<Mat> mags = vector<Mat>());

