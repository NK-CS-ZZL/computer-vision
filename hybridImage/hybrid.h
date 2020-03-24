#pragma once
#include <opencv2/opencv.hpp>

using cv::Rect;


bool hybridGray(const char* highPass, const char* lowPass, const double percent = 0.125, Rect range1 = Rect(), Rect range2 = Rect());

bool hybridColor(const char* highPass, const char* lowPass, const double percent = 0.125, Rect range1 = Rect(), Rect range2 = Rect());