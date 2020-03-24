#include "utilities.h"
#include "hybrid.h"

using namespace std;
using namespace cv;

// remove a region in "src". Flag == INTERNAL mean removes pixel in "range".
// Flag == EXTERNAL mean removes pixel out of "range".
void imgCut(const Mat& src, Mat& dst, Rect range, cutFlag flag) {
	Mat mask;
	if (flag == cutFlag::INTERNAL)
	{
		mask = Mat::zeros(src.size(), CV_8UC1);
		mask(range).setTo(255);
	}
	else if (flag == cutFlag::EXTERNAL) {
		mask = Mat(src.rows, src.cols, CV_8UC1, Scalar(255));
		mask(range).setTo(0);
	}
	src.copyTo(dst);
	dst.setTo(0, mask);
}

// a wrapper of dft
void dft(const Mat& src, Mat& dst) {
	int oldRows = src.rows, oldCols = src.cols;
	int newRows = getOptimalDFTSize(oldRows);
	int newCols = getOptimalDFTSize(oldCols);
	Mat tmp;
	copyMakeBorder(src, tmp, 0, newRows - oldRows, 0, newCols - oldCols, BORDER_CONSTANT, Scalar(0));
	vector<Mat> vecMatFourior = { Mat_<float>(tmp), Mat::zeros(tmp.size(), CV_32F) };
	Mat mSrc;
	merge(vecMatFourior, mSrc);
	cv::dft(mSrc, dst);
}

// a wrapper of idft
void idft(const Mat& src, Mat& dst) {
	cv::idft(src, dst, DFT_SCALE | DFT_REAL_OUTPUT);
}

// Using frequency matrix given by dft calculates magnitude.
Mat magnitude(const Mat& mat) {
	vector<Mat> channels;
	split(mat, channels);

	Mat& mReal = channels[0];
	Mat& mImaginary = channels[1];

	Mat tmp;
	magnitude(mReal, mImaginary, tmp);

	return tmp;
}

// Log and normalization.
Mat logAndNormal(const Mat& mat) {
	Mat tmp;
	if (mat.channels() == 1)
		tmp = mat + Scalar(1);
	else if (mat.channels() == 3)
		tmp = mat + Scalar(1, 1, 1);
	log(tmp, tmp);
	normalize(tmp, tmp, 0, 255, NORM_MINMAX);
	return tmp;
}

// Swap the first quadrant and the third.
// Swap the second quadrant and the fourth.
// To make all low-frequency signals converge in the center.
void diagonalExchange(Mat& mat) {
	Mat mQuadrant1 = mat(Rect(mat.cols / 2, 0, mat.cols / 2, mat.rows / 2));
	Mat mQuadrant2 = mat(Rect(0, 0, mat.cols / 2, mat.rows / 2));
	Mat mQuadrant3 = mat(Rect(0, mat.rows / 2, mat.cols / 2,mat.rows / 2));
	Mat mQuadrant4 = mat(Rect(mat.cols / 2, mat.rows / 2, mat.cols / 2, mat.rows / 2));

	Mat mChange1 = mQuadrant1.clone();
	mQuadrant3.copyTo(mQuadrant1);
	mChange1.copyTo(mQuadrant3);

	Mat mChange2 = mQuadrant2.clone();
	mQuadrant4.copyTo(mQuadrant2);
	mChange2.copyTo(mQuadrant4);
}

void highPassFliter(const Mat& src, Mat& dst, const double percent, vector<Mat>* mag) {
	int rows = src.rows, cols = src.cols;
	int inCols = int(percent * cols), inRows = int(percent * rows);
	int exCols = int((1 - percent) / 2 * cols), exRows = int((1 - percent) / 2 * rows);
	Rect range(exCols, exRows, inCols, inRows);
	src.copyTo(dst);
	// Make all low-frequency signals converge in the center.
	diagonalExchange(dst);
	if (mag != nullptr)
		mag->push_back(logAndNormal(magnitude(dst)));
	// Delete them.
	imgCut(dst, dst, range, cutFlag::INTERNAL);
	if (mag != nullptr)
		mag->push_back(logAndNormal(magnitude(dst)));
	// Restore.
	diagonalExchange(dst);
}

void lowPassFliter(const Mat& src, Mat& dst, const double percent, vector<Mat>* mag) {
	int rows = src.rows, cols = src.cols;
	int inCols = int(percent * cols), inRows = int(percent * rows);
	int exCols = int((1 - percent) / 2 * cols), exRows = int((1 - percent) / 2 * rows);
	Rect range(exCols, exRows, inCols, inRows);
	src.copyTo(dst);
	// Make all low-frequency signals converge in the center.
	diagonalExchange(dst);
	if (mag != nullptr)
		mag->push_back(logAndNormal(magnitude(dst)));
	// Only remain low-frequency signal.
	imgCut(dst, dst, range, cutFlag::EXTERNAL);
	if (mag != nullptr)
		mag->push_back(logAndNormal(magnitude(dst)));
	// Restore.
	diagonalExchange(dst);
}

// If you want get magnitude images, transform a non-null pointer pointed to a vector<Mat>.
void _hybrid(const Mat& mHp, const Mat& mLp, Mat& res, const double percent, vector<Mat>* mag) {
	Mat mFreqHp, mFreqLp;
	dft(mHp, mFreqHp);
	dft(mLp, mFreqLp);
	if (mag != nullptr) {
		mag->clear();
	}
	highPassFliter(mFreqHp, mFreqHp, percent, mag);
	lowPassFliter(mFreqLp, mFreqLp, percent, mag);
	idft(mFreqHp, mFreqHp);
	idft(mFreqLp, mFreqLp);
	addWeighted(mFreqHp, 1, mFreqLp, 1, 0, mFreqHp);
	normalize(mFreqHp, mFreqHp, 0, 255, NORM_MINMAX);
	mFreqHp.convertTo(res, CV_8U);
}

// splicing multiple pictures into a picture. rowNum and colNum indicates how many sub picture
// in a single row or colomn. newSize indicates the size of sub pictures.
Mat combine(const vector<Mat>& imgs, int rowNum, int colNum, Size newSize) {
	size_t imgNum = imgs.size();
	const int margin = 0;
	vector<Mat> resized(imgNum);
	for (int i = 0; i < imgNum; i++) {
		resize(imgs[i], resized[i], newSize);
	}
	int newWidth = (colNum - 1) * margin + colNum * newSize.width;
	int newHeight = (rowNum - 1) * margin + rowNum * newSize.height;
	Mat res;
	if (imgs[0].channels() == 1)
		res = Mat(newHeight, newWidth, CV_8UC1, Scalar(255));
	else if (imgs[0].channels() == 3)
		res = Mat(newHeight, newWidth, CV_8UC3, Scalar(255, 255, 255));
	else
		return Mat();

	int x = 0, y = 0, count = 0;
	while (count < imgNum) {
		Mat region = res(Rect(x * (newSize.width + margin), y * (newSize.height + margin), newSize.width, newSize.height));
		resized[count].copyTo(region);
		count++;
		if (x == colNum - 1) { x = 0; y++; }
		else { x++; }
	}
	return res;
}

// Arrange result pictures into a proper order and call combine() function.
// Then it shows all source pictures and result pictures.
// You can choose whether display magnitude image or not.
void formatDisplay(const vector<Mat>& srcs, const Mat& res, const vector<Mat> mags) {
	Mat combImg, combMag;
	vector<Mat> tmp;
	tmp.push_back(srcs[0]);
	tmp.push_back(res(Rect(0, 0, res.cols / 2, res.rows / 2)));
	tmp.push_back(res(Rect(res.cols / 2, 0, res.cols / 2, res.rows / 2)));
	tmp.push_back(srcs[1]);
	tmp.push_back(res(Rect(0, res.rows / 2, res.cols / 2, res.rows / 2)));
	tmp.push_back(res(Rect(res.cols / 2, res.rows / 2, res.cols / 2, res.rows / 2)));
	combImg = combine(tmp, 2, 3, srcs[0].size() / 2);
	imwrite("result1.jpg", combImg);
	if (!mags.empty())
	{
		tmp.clear();
		tmp.push_back(srcs[0]);
		tmp.push_back(logAndNormal(mags[0]));
		tmp.push_back(logAndNormal(mags[1]));
		tmp.push_back(srcs[1]);
		tmp.push_back(logAndNormal(mags[2]));
		tmp.push_back(logAndNormal(mags[3]));
		// data type of "mags" is float, we must convert it into uchar to display.
		if (srcs[0].channels() == 1) {
			for (Mat& m : tmp) {
				m.convertTo(m, CV_8UC1);
			}
		}
		else if (srcs[0].channels() == 3) {
			for (Mat& m : tmp) {
				m.convertTo(m, CV_8UC3);
			}
		}
		combMag = combine(tmp, 2, 3, srcs[0].size() / 2);
		imwrite("result2.jpg", combMag);
		imshow("Magnitude", combMag);
	}
	imshow("Images", combImg);	
	waitKey();
}

// Hybrid for grayscale/singal-channel image.
// "percent" indicates what percentage of low-frequency singals you want to cut.
// hpRoi/lpRoi indicates region of interest.
bool hybridGray(const char* highPass, const char* lowPass, const double percent, Rect hpRoi, Rect lpRoi) {
	Mat hpMat = imread(highPass, 0);
	Mat lpMat = imread(lowPass, 0);
	if (hpMat.empty() || lpMat.empty())
		return false;
	if (!hpRoi.empty()) {
		hpMat = hpMat(hpRoi);
	}
	if (!lpRoi.empty()) {
		lpMat = lpMat(lpRoi);
	}
	resize(lpMat, lpMat, hpMat.size());
	Mat res;
	vector<Mat> imgs, mags;
	imgs.push_back(hpMat);
	imgs.push_back(lpMat);
	_hybrid(hpMat, lpMat, res, percent, &mags);
	res = res(Rect(0, 0, hpMat.cols, hpMat.rows));
	formatDisplay(imgs, res, mags);
	return true;
}

// Hybrid for colored/multi-channel image.
// For each channel, call hybrid function for singal channel and merge them.
bool hybridColor(const char* highPass, const char* lowPass, const double percent, Rect hpRoi, Rect lpRoi) {
	Mat hpMat = imread(highPass, 1);
	Mat lpMat = imread(lowPass, 1);
	int channels = hpMat.channels();
	if (hpMat.empty() || lpMat.empty())
		return false;
	if (!hpRoi.empty()) {
		hpMat = hpMat(hpRoi);
	}
	if (!lpRoi.empty()) {
		lpMat = lpMat(lpRoi);
	}
	resize(lpMat, lpMat, hpMat.size());

	Mat res;
	vector<Mat> mags(4), imgs;
	vector<Mat> hpVec, lpVec, resVec(channels);
	split(hpMat, hpVec);
	split(lpMat, lpVec);

	vector<vector<Mat>> magsVec(channels);

	for (int i = 0; i < hpVec.size(); i++) {
		_hybrid(hpVec[i], lpVec[i], resVec[i], percent, &(magsVec[i]));
	}
	merge(resVec, res);

	for (int i = 0; i < 4; i++) {
		vector<Mat> tmp;
		for (int j = 0; j < channels; j++) {
			tmp.push_back(magsVec[j][i]);
		}
		merge(tmp, mags[i]);
	}

	imgs.push_back(hpMat);
	imgs.push_back(lpMat);
	formatDisplay(imgs, res, mags);
	return true;
}