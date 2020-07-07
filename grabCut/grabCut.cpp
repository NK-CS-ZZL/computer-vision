#include "grabCut.h"
#include <limits>
// #define DEBUG
#ifdef DEBUG
#include <fstream>
#include <iostream>
using namespace std;
using namespace cv;
#endif

using std::vector;
using std::numeric_limits;

static const int BGD = 0;
static const int OBJ = 1;
static const int MAYBE_BGD = 2;
static const int MAYBE_OBJ = 3;

void GrabCut::setInitMatte(const Mat& trimap) {
	int rows = trimap.rows, cols = trimap.cols;
	Point p;
	matte.create(rows, cols, CV_8UC1);
	for (p.y = 0; p.y < rows; p.y++) {
		for (p.x = 0; p.x < cols; p.x++) {
			if (trimap.at<uchar>(p) == 0) {
				matte.at<uchar>(p) = BGD;
			}
			else if (trimap.at<uchar>(p) == 255) {
				matte.at<uchar>(p) = OBJ;
			}
			else {
				matte.at<uchar>(p) = MAYBE_OBJ;
			}
		}
	}
#ifdef DEBUG

	ofstream fout("trimap.txt");
	for (int i = 0; i < trimap.rows; i++)
	{
		for (int j = 0; j < trimap.cols; j++)
		{
			fout << (int)trimap.ptr<uchar>(i)[j] << '\t';
		}
		fout << endl;
	}
	fout.close();
	imshow("tri2", trimap);
	waitKey();
	cout << beta << endl;
#endif
}

double GrabCut::calBeta(const Mat& input) {
	double beta = 0.0f;
	double total = 0.0;
	int rows = input.rows, cols = input.cols;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			// 八个方向，每个像素计算四个方向即可完全遍历
			Vec3d curr = (Vec3d)input.at<Vec3b>(r, c);
			// 上方
			if (r > 0) {
				Vec3d adjacent = (Vec3d)input.at<Vec3b>(r - 1, c);
				Vec3d diff = curr - adjacent;
				total += diff.dot(diff);
			}
			// 左侧
			if (c > 0) {
				Vec3d adjacent = (Vec3d)input.at<Vec3b>(r, c - 1);
				Vec3d diff = curr - adjacent;
				total += diff.dot(diff);
			}
			// 左上
			if (r > 0 && c > 0) {
				Vec3d adjacent = (Vec3d)input.at<Vec3b>(r - 1, c - 1);
				Vec3d diff = curr - adjacent;
				total += diff.dot(diff);
			}
			// 右上
			if (r > 0 && c < cols - 1) {
				Vec3d adjacent = (Vec3d)input.at<Vec3b>(r - 1, c + 1);
				Vec3d diff = curr - adjacent;
				total += diff.dot(diff);
			}
		}
	}
	if (total < numeric_limits<double>::epsilon()) beta = 0;
	else beta = 1.0 / (2 * total / (4 * rows * cols - 3 * rows - 3 * cols + 2));
	return beta;

}

void GrabCut::calNWeight() {
	const double diagDis = gamma / sqrt(2.0);
	int rows = img.rows, cols = img.cols;
	

	leftW.create(img.rows, img.cols, CV_64FC1);
	upleftW.create(img.rows, img.cols, CV_64FC1);
	upW.create(img.rows, img.cols, CV_64FC1);
	uprightW.create(img.rows, img.cols, CV_64FC1);

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			Vec3d color = img.at<Vec3b>(r, c);
			if (c - 1 >= 0) // left 
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(r, c - 1);
				leftW.at<double>(r, c) = gamma * exp(-beta * diff.dot(diff));
			}
			else
				leftW.at<double>(r, c) = 0;

			if (c - 1 >= 0 && r - 1 >= 0) // upleft  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(r - 1, c - 1);
				upleftW.at<double>(r, c) = diagDis * exp(-beta * diff.dot(diff));
			}
			else
				upleftW.at<double>(r, c) = 0;

			if (r - 1 >= 0) // up  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(r - 1, c);
				upW.at<double>(r, c) = gamma * exp(-beta * diff.dot(diff));
			}
			else
				upW.at<double>(r, c) = 0;

			if (c + 1 < cols && r - 1 >= 0) // upright  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(r - 1, c + 1);
				uprightW.at<double>(r, c) = diagDis * exp(-beta * diff.dot(diff));
			}
			else
				uprightW.at<double>(r, c) = 0;

		}
	}
}

void GrabCut::initGMMmodel() {
	const int kmeansItCount = 10;
	Mat bgdLabels, objLabels;
	vector<Vec3f> bgdSamples, objSamples;
	int rows = img.rows, cols = img.cols;
	Point p;
	for (p.y = 0; p.y < rows; p.y++) {
		for (p.x = 0; p.x < cols; p.x++) {
			if (matte.at<uchar>(p) == BGD || matte.at<uchar>(p) == MAYBE_BGD) {
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
			}
			else {
				objSamples.push_back((Vec3f)img.at<Vec3b>(p));
			}
		}
	}
	Mat _bgdSample((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	kmeans(_bgdSample, GMM::K, bgdLabels, TermCriteria(TermCriteria::Type::COUNT, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);
	Mat _objSample((int)objSamples.size(), 3, CV_32FC1, &objSamples[0][0]);
	kmeans(_objSample, GMM::K, objLabels, TermCriteria(TermCriteria::Type::COUNT, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);

	bgdGMM.initLearning();
	for (int i = 0; i < bgdSamples.size(); i++) {
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	}
	bgdGMM.endLearning();

	objGMM.initLearning();
	for (int i = 0; i < objSamples.size(); i++) {
		objGMM.addSample(objLabels.at<int>(i, 0), objSamples[i]);
	}
	objGMM.endLearning();
}

void GrabCut::assignGMM() {
	Point p;
	int rows = img.rows, cols = img.cols;
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			Vec3d color = (Vec3d)img.at<Vec3b>(p);
			uchar t = matte.at<uchar>(p);
			if (t == BGD || t == MAYBE_BGD) {
				idxs.at<uchar>(p) = bgdGMM.whichComponent(color);
			}
			else {
				idxs.at<uchar>(p) = objGMM.whichComponent(color);
			}
		}
	}
}

void GrabCut::learnGMM() {
	bgdGMM.initLearning();
	objGMM.initLearning();
	Point p;
	int rows = img.rows, cols = img.cols;
	for (int i = 0; i < GMM::K; i++) {
		for (p.y = 0; p.y < rows; p.y++) {
			for (p.x = 0; p.x < cols; p.x++) {
				int idx = (int)idxs.at<uchar>(p);
				if (idx == i) {
					if (matte.at<uchar>(p) == BGD || matte.at<uchar>(p) == MAYBE_BGD) {
						bgdGMM.addSample(idx, img.at<Vec3b>(p));
					}
					else {
						objGMM.addSample(idx, img.at<Vec3b>(p));
					}
				}
			}
		}
	}
	bgdGMM.endLearning();
	objGMM.endLearning();
}

void GrabCut::getGraph() {
	int rows = img.rows, cols = img.cols;
	int nCount = cols * rows, eCount = 2 * (4 * nCount - 3 * cols - 3 * rows + 2);
	GraphType* g = new GraphType(/*estimated # of nodes*/ nCount, /*estimated # of edges*/ eCount);
	Point p;
	for (p.y = 0; p.y < rows; p.y++) {
		for (p.x = 0; p.x < cols; p.x++) {
			int nodeID = g->add_node();
			Vec3d color = (Vec3d)img.at<Vec3b>(p);
			double wSource = 0.0, wSink = 0.0;
			if (matte.at<uchar>(p) == MAYBE_BGD || matte.at<uchar>(p) == MAYBE_OBJ) {
				wSource = -log(bgdGMM(color));
				wSink = -log(objGMM(color));
			}
			else if (matte.at<uchar>(p) == BGD) {
				wSink = lambda;
			}
			else {
				wSource = lambda;
			}
			g->add_tweights(nodeID, wSource, wSink);
			if (p.x > 0) {
				// 与左侧node相连
				double w = leftW.at<double>(p);
				g->add_edge(nodeID, nodeID - 1, w, w);
			}
			if (p.y > 0) {
				double w = upW.at<double>(p);
				g->add_edge(nodeID, nodeID - cols, w, w);
			}
			if (p.x > 0 && p.y > 0) {
				double w = upleftW.at<double>(p);
				g->add_edge(nodeID, nodeID - cols - 1, w, w);
			}
			if (p.x < cols - 1 && p.y > 0) {
				double w = uprightW.at<double>(p);
				g->add_edge(nodeID, nodeID - cols + 1, w, w);
			}
		}
	}
	graph = g;
}

void GrabCut::graphSegment() {
	graph->maxflow();
	Point p;
	int rows = img.rows, cols = img.cols;
	for (p.y = 0; p.y < rows; p.y++) {
		for (p.x = 0; p.x < cols; p.x++) {
			if (matte.at<uchar>(p) == MAYBE_BGD || matte.at<uchar>(p) == MAYBE_OBJ) {
				if (graph->what_segment(p.y * cols + p.x) == GraphType::SOURCE) {
					matte.at<uchar>(p) = MAYBE_OBJ;
				}
				else {
					matte.at<uchar>(p) = MAYBE_BGD;
				}
			}
		}
	}
	delete graph;
}

void GrabCut::init(const Mat& input, bool isTest) {
	input.copyTo(img);
	if(!isTest)
		this->matte = drawRect(img) / 255 * 3;
	this->beta = this->calBeta(input);
	this->idxs.create(img.rows, img.cols, CV_8UC1);
#ifdef DEBUG
	ofstream fout("matte.txt");
	for (int i = 0; i < matte.rows; i++)
	{
		for (int j = 0; j < matte.cols; j++)
		{
			fout << (int)matte.ptr<uchar>(i)[j] << '\t';
		}
		fout << endl;
	}
	fout.close();
	cout << beta << endl;
#endif
}

void GrabCut::run() {
	if (img.empty()) {
		CV_Assert(true);
	}
	initGMMmodel();
	calNWeight();
	for (int i = 0; i < iterTimes; i++) {
		assignGMM();
		learnGMM();
		getGraph();
		graphSegment();
	}
#ifdef DEBUG
	ofstream fout("matteEnd.txt");
	for (int i = 0; i < matte.rows; i++)
	{
		for (int j = 0; j < matte.cols; j++)
		{
			fout << (int)matte.ptr<uchar>(i)[j] << '\t';
		}
		fout << endl;
	}
	fout.close();
#endif
}

Mat GrabCut::getMatte() {
	Point p;
	int rows = img.rows, cols = img.cols;
	Mat result(rows, cols, CV_8UC1);
	for (p.y = 0; p.y < rows; p.y++) {
		for (p.x = 0; p.x < cols; p.x++) {
			if (matte.at<uchar>(p) == MAYBE_OBJ || matte.at<uchar>(p) == OBJ) {
				result.at<uchar>(p) = 255;
			}
			else {
				result.at<uchar>(p) = 0;
			}
		}
	}
	return result;
}

void GrabCut::interact() {
	drawLine(img, matte);
	run();
}