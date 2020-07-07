#include "GMM.h"
using namespace cv;


void GMM::calInverseCovAndDet(int i) {
	if (coefs[i] > 0) {
		double* c = covs + 9 * i;
		double det = detCov[i] 
					=c[0] * (c[4] * c[8] - c[5] * c[7])
					- c[1] * (c[3] * c[8] - c[5] * c[6])
					+ c[2] * (c[3] * c[7] - c[4] * c[6]);
		// A^(-1) = A*/|A|
		inverseCovs[i][0][0] = (c[4] * c[8] - c[5] * c[7]) / det;
		inverseCovs[i][1][0] = -(c[3] * c[8] - c[5] * c[6]) / det;
		inverseCovs[i][2][0] = (c[3] * c[7] - c[4] * c[6]) / det;
		inverseCovs[i][0][1] = -(c[1] * c[8] - c[2] * c[7]) / det;
		inverseCovs[i][1][1] = (c[0] * c[8] - c[2] * c[6]) / det;
		inverseCovs[i][2][1] = -(c[0] * c[7] - c[1] * c[6]) / det;
		inverseCovs[i][0][2] = (c[1] * c[5] - c[2] * c[4]) / det;
		inverseCovs[i][1][2] = -(c[0] * c[5] - c[2] * c[3]) / det;
		inverseCovs[i][2][2] = (c[0] * c[4] - c[1] * c[3]) / det;
	}
}

GMM::GMM() {
	Mat m;
	m.create(1, modelSize * K, CV_64FC1);
	m.setTo(Scalar(0));
	this->model = m;
	coefs = model.ptr<double>(0);
	means = coefs + K;
	covs = means + K * 3;
	for (int i = 0; i < K; i++) {
		if (coefs[i] > 0) {
			calInverseCovAndDet(i);
		}
	}
	totalSampleCount = 0;
}

GMM::GMM(Mat& m) {
	if (m.empty()) {
		m.create(1, modelSize * K, CV_64FC1);
		m.setTo(Scalar(0));
	}
	else if ((m.type() != CV_64FC1) || (m.rows != 1) || (m.cols != modelSize * K)) {
		CV_Assert(true);
	}
	this->model = m;
	coefs = model.ptr<double>(0);
	means = coefs + K;
	covs = means + K * 3;
	for (int i = 0; i < K; i++) {
		if (coefs[i] > 0) {
			calInverseCovAndDet(i);
		}
	}
	totalSampleCount = 0;
}

double GMM::operator()(const Vec3d color) const {
	double prob = 0.0;
	for (int i = 0; i < K; i++) {
		prob += this->operator()(i, color) * coefs[i];
	}
	return prob;
}

double GMM::operator()(int i, const Vec3d color) const {
	double prob = 0.0;
	if (coefs[i] > 0) {
		Vec3d diff = color;
		double* m = means + 3 * i;
		diff[0] -= m[0];
		diff[1] -= m[1];
		diff[2] -= m[2];
		double mult = diff[0] * (diff[0] * inverseCovs[i][0][0] + diff[1] * inverseCovs[i][1][0] + diff[2] * inverseCovs[i][2][0])
					+ diff[1] * (diff[0] * inverseCovs[i][0][1] + diff[1] * inverseCovs[i][1][1] + diff[2] * inverseCovs[i][2][1])
					+ diff[2] * (diff[0] * inverseCovs[i][0][2] + diff[1] * inverseCovs[i][1][2] + diff[2] * inverseCovs[i][2][2]);
		prob = 1.0 / sqrt(detCov[i]) * exp(-0.5 * mult);
	}
	return prob;
}

int GMM::whichComponent(const Vec3d color) const {
	int k = 0;
	double max = 0.0;
	for (int i = 0; i < K; i++) {
		double prob = this->operator()(i, color);
		if (prob > max) {
			k = i;
			max = prob;
		}
	}
	return k;
}

void GMM::initLearning() {
	for (int i = 0; i < K; i++) {
		sums[i][0] = sums[i][1] = sums[i][2] = 0.0;
		prods[i][0][0] = prods[i][0][1] = prods[i][0][2] = 0;
		prods[i][1][0] = prods[i][1][1] = prods[i][1][2] = 0;
		prods[i][2][0] = prods[i][2][1] = prods[i][2][2] = 0;
		sampleCounts[i] = 0;
	}
	totalSampleCount = 0;
}

void GMM::addSample(int i, const Vec3d color) {
	sums[i][0] += color[0];
	sums[i][1] += color[1];
	sums[i][2] += color[2];
	prods[i][0][0] += color[0] * color[0];
	prods[i][1][1] += color[1] * color[1];
	prods[i][2][2] += color[2] * color[2];

	prods[i][0][1] += color[0] * color[1];
	prods[i][1][0] += color[1] * color[0];
	prods[i][0][2] += color[0] * color[2];
	prods[i][2][0] += color[2] * color[0];
	prods[i][1][2] += color[1] * color[2];
	prods[i][2][1] += color[2] * color[1];

	sampleCounts[i]++;
	totalSampleCount++;
}

void GMM::endLearning() {
	const double epsilon = 1e-2;
	for (int i = 0; i < K; i++) {
		int n = sampleCounts[i];
		if (n == 0) {
			coefs[i] = 0;
		}
		else {
			coefs[i] =  (double)n / totalSampleCount;
			double* m = means + 3 * i;
			m[0] = sums[i][0] / n; m[1] = sums[i][1] / n; m[2] = sums[i][2] / n;
			double* c = covs + 9 * i;
			c[0] = prods[i][0][0] / n - m[0] * m[0]; c[1] = prods[i][0][1] / n - m[0] * m[1]; c[2] = prods[i][0][2] / n - m[0] * m[2];
			c[3] = prods[i][1][0] / n - m[1] * m[0]; c[4] = prods[i][1][1] / n - m[1] * m[1]; c[5] = prods[i][1][2] / n - m[1] * m[2];
			c[6] = prods[i][2][0] / n - m[2] * m[0]; c[7] = prods[i][2][1] / n - m[2] * m[1]; c[8] = prods[i][2][2] / n - m[2] * m[2];
			double det = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
			
			// ´¦ÀíÆæÒì¾ØÕó
			if (det <= std::numeric_limits<double>::epsilon()) {
				c[0] += epsilon;
				c[4] += epsilon;
				c[8] += epsilon;
			}
			calInverseCovAndDet(i);
		}
	}
}