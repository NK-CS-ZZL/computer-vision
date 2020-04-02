#pragma once
#include <opencv2/opencv.hpp>
using cv::Mat;

enum class normMode { L1, L2 };
void canny(const Mat& src, Mat& dst, double TL, double TH, int kSize = 3, normMode mode = normMode::L1);