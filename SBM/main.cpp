#include "SemiGlobalMatching.h"
#include <opencv2/opencv.hpp>




int main(int argv, char** argc)
{


	std::string path_left = "D:/Data/im2.png";
	std::string path_right = "D:/Data/im6.png";

	cv::Mat img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
	cv::Mat img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

	const int32_t width = static_cast<uint32_t>(img_left.cols);
	const int32_t height = static_cast<uint32_t>(img_right.rows);

	unsigned char* bytes_left = new unsigned char[width * height];
	unsigned char* bytes_right = new unsigned char[width * height];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			bytes_left[i * width + j] = img_left.at<unsigned char>(i, j);
			bytes_right[i * width + j] = img_right.at<unsigned char>(i, j);
		}
	}



	SemiGlobalMatching::SGMOption sgm_option;

	sgm_option.num_paths = 8;

	sgm_option.min_disparity = 0;
	sgm_option.max_disparity = 64;

	sgm_option.is_check_lr = true;
	sgm_option.lrcheck_thres = 1.0f;

	sgm_option.is_check_unique = true;
	sgm_option.uniqueness_ratio = 0.99;

	sgm_option.is_remove_speckles = true;
	sgm_option.min_speckle_aera = 30;

	sgm_option.p1 = 10;
	sgm_option.p2_init = 150;

	sgm_option.is_fill_holes = true;


	SemiGlobalMatching sgm;


	sgm.Initialize(width, height, sgm_option);


	float* disparity = new float[uint32_t(width * height)]();
	sgm.Match(bytes_left, bytes_right, disparity);


	cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	float min_disp = width, max_disp = -width;
	for (int32_t i = 0; i < height; i++) {
		for (int32_t j = 0; j < width; j++) {
			const float disp = disparity[i * width + j];
			if (disp != INVALID_FLOAT) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (int32_t i = 0; i < height; i++) {
		for (int32_t j = 0; j < width; j++) {
			const float disp = disparity[i * width + j];
			if (disp == INVALID_FLOAT) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}

	cv::imshow(" ”≤ÓÕº", disp_mat);
	cv::Mat disp_color;
	cv::imwrite("res2.jpg", disp_mat);
	cv::waitKey(0);
	return 0;
}