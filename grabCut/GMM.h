#pragma once
#include <opencv2/opencv.hpp>
using cv::Mat;
using cv::Vec3d;

class GMM
{
public:
	static const int K = 5;
private:
	static const int modelSize = 13;
	Mat model;
	double* coefs;
	double* means;
	double* covs;

	double inverseCovs[K][3][3];	// Э�������
	double detCov[K];				//	Э���������ʽ
	double sums[K][3];				// ���ڼ���means
	double prods[K][3][3];			// ���ڼ���covs
	int sampleCounts[K];
	int totalSampleCount;

	void calInverseCovAndDet(int i);

public:
	GMM();
	GMM(Mat& m);
	double operator()(const Vec3d color) const;
	double operator()(int i, const Vec3d color) const;
	int whichComponent(const Vec3d color) const;
	void initLearning();
	void addSample(int ci, const Vec3d color);
	void endLearning();
};

