#pragma once
#include "interact.h"
#include "GMM.h"
#include "graph.h"

using namespace cv;

typedef Graph<double, double, double> GraphType;

class GrabCut {
	const double gamma = 50;
	const double lambda = 9 * gamma;
	const int iterTimes = 2;
	Mat img;
	Mat matte;
	Mat idxs;	// uninitialized
	double beta;

	Mat leftW;
	Mat upleftW;
	Mat upW;
	Mat uprightW;

	GMM bgdGMM;
	GMM objGMM;

	GraphType* graph;

	void calNWeight();

	double calBeta(const Mat& input);
	void initGMMmodel();
	void assignGMM();
	void learnGMM();
	void getGraph();
	void graphSegment();
public:
	GrabCut() : beta(0.0f), graph(nullptr) {}
	void init(const Mat& input, bool isTest = false);
	void setInitMatte(const Mat& trimap);
	void run();
	void interact();
	Mat getMatte();
};