#include "interact.h"
#include <string>
// #define DEBUG

using namespace std;
using namespace cv;

static const int BGD = 0;
static const int OBJ = 1;
static const int MAYBE_BGD = 2;
static const int MAYBE_OBJ = 3;

void lineCallBack(int event, int x, int y, int flags, void* param);
void rectCallBack(int event, int x, int y, int flags, void* param);

struct Params {
	Mat& img;
	Mat& mask;
	Params(Mat& image, Mat& msk) : img(image), mask(msk) {}
};

Mat drawLine(const Mat& img, Mat& mask) {
	const string imgWinName = "imgWindows";
	const string maskWinName = "maskWindows";

	Mat tmp;
	if (mask.empty()) {
		mask = Mat(img.size().height, img.size().width, CV_8UC1);
		mask = Scalar::all(0);
	}
	img.copyTo(tmp);

	namedWindow(imgWinName);
#ifdef DEBUG
	namedWindow(maskWinName);
#endif

	Params params(tmp, mask);

	setMouseCallback(imgWinName, lineCallBack, (void*)&params);

	while (1) {
		imshow(imgWinName, tmp);
#ifdef DEBUG
		imshow(maskWinName, mask);
#endif

		if (waitKey(10) == 'q')
		{
			destroyWindow(imgWinName);
#ifdef DEBUG
			destroyWindow(maskWinName);
#endif
			break;
		}
	}
	return mask;
}

Mat drawRect(const Mat& img) {
	const string imgWinName = "imgWindows";
	const string maskWinName = "maskWindows";

	Mat mask = Mat(img.size().height, img.size().width, CV_8UC1);
	Mat tmp;
	mask = Scalar::all(0);

	img.copyTo(tmp);
	namedWindow(imgWinName);
#ifdef DEBUG
	namedWindow(maskWinName);
#endif
	Params params(tmp, mask);

	setMouseCallback(imgWinName, rectCallBack, (void*)&params);

	while (1) {
		imshow(imgWinName, tmp);
#ifdef DEBUG
		imshow(maskWinName, mask);
#endif

		if (waitKey(10) == 'q')
		{
			destroyWindow(imgWinName);
#ifdef DEBUG
			destroyWindow(maskWinName);
#endif
			break;
		}
	}
	return mask;
}

void lineCallBack(int event, int x, int y, int flags, void* param) {
	Params& params = *(Params*)param;
	Mat& img = params.img;
	Mat& mask = params.mask;

	static bool leftDown = false;
	static bool rightDown = false;
	static Point currPoint, prevPoint;

	switch (event) {
	case EVENT_MOUSEMOVE: {
#ifdef DEBUG
		printf("mouse moves\n");
#endif
		if (leftDown == true) {
#ifdef DEBUG
			printf("draw\n");
#endif // DEBUG

			currPoint = Point(x, y);

			line(img, prevPoint, currPoint, Scalar(255, 0, 0), 4, 4);
			line(mask, prevPoint, currPoint, Scalar(OBJ), 4, 4);

			prevPoint = currPoint;
		}
		if (rightDown == true) {
			currPoint = Point(x, y);
			line(img, prevPoint, currPoint, Scalar(0, 255, 0), 4, 4);
			line(mask, prevPoint, currPoint, Scalar(BGD), 4, 4);
			prevPoint = currPoint;
		}
		break;
	}
	case EVENT_LBUTTONDOWN:
	{
#ifdef DEBUG
		printf("left down\n");
#endif // DEBUG
		leftDown = true;
		prevPoint = Point(x, y);
		break;
	}
	case EVENT_LBUTTONUP:
	{
#ifdef DEBUG
		printf("left up\n");
#endif // DEBUG
		leftDown = false;
		break;
	}
	case EVENT_RBUTTONDOWN:
	{
		rightDown = true;
		prevPoint = Point(x, y);
		break;
	}
	case EVENT_RBUTTONUP:
	{
		rightDown = false;
		break;
	}
	}
}

void rectCallBack(int event, int x, int y, int flags, void* param) {
	static bool leftDown = false;
	static Point start;
	Params& params = *(Params*)param;
	Mat& img = params.img;
	Mat& mask = params.mask;
	switch (event) {
	case EVENT_LBUTTONDOWN: {
#ifdef DEBUG
		printf("left down\n");
#endif // DEBUG
		start = Point(x, y);
		break;
	}
	case EVENT_LBUTTONUP:
	{
#ifdef DEBUG
		printf("left up\n");
#endif // DEBUG
		Point end(x, y);
		rectangle(img, start, end, Scalar(255, 0, 0));
		rectangle(mask, start, end, Scalar(255), FILLED);
		break;
	}
	}
}

