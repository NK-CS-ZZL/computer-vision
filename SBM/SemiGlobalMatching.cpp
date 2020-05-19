#include "SemiGlobalMatching.h"
#include <algorithm>
#include <vector>
#include <cassert>

#ifndef SAFE_DELETE
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}
#endif

SemiGlobalMatching::SemiGlobalMatching() : width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
census_left_(nullptr), census_right_(nullptr),
cost_init_(nullptr), cost_aggr_(nullptr),
cost_aggr_1_(nullptr), cost_aggr_2_(nullptr),
cost_aggr_3_(nullptr), cost_aggr_4_(nullptr),
cost_aggr_5_(nullptr), cost_aggr_6_(nullptr),
cost_aggr_7_(nullptr), cost_aggr_8_(nullptr),
disp_left_(nullptr), disp_right_(nullptr),
is_initialized_(false)
{
}


SemiGlobalMatching::~SemiGlobalMatching()
{
	Release();
	is_initialized_ = false;
}

bool SemiGlobalMatching::Initialize(const int32_t& width, const int32_t& height, const SGMOption& option)
{

	width_ = width;
	height_ = height;
	option_ = option;

	if (width == 0 || height == 0) {
		return false;
	}


	const int32_t img_size = width * height;
	census_left_ = new uint32_t[img_size]();
	census_right_ = new uint32_t[img_size]();


	const int32_t disp_range = option.max_disparity - option.min_disparity;
	if (disp_range <= 0) {
		return false;
	}

	const int32_t size = width * height * disp_range;
	cost_init_ = new uint8_t[size]();
	cost_aggr_ = new uint16_t[size]();
	cost_aggr_1_ = new uint8_t[size]();
	cost_aggr_2_ = new uint8_t[size]();
	cost_aggr_3_ = new uint8_t[size]();
	cost_aggr_4_ = new uint8_t[size]();
	cost_aggr_5_ = new uint8_t[size]();
	cost_aggr_6_ = new uint8_t[size]();
	cost_aggr_7_ = new uint8_t[size]();
	cost_aggr_8_ = new uint8_t[size]();

	disp_left_ = new float[img_size]();
	disp_right_ = new float[img_size]();

	is_initialized_ = census_left_ && census_right_ && cost_init_ && cost_aggr_ && disp_left_;

	return is_initialized_;
}

void SemiGlobalMatching::Release()
{
	SAFE_DELETE(census_left_);
	SAFE_DELETE(census_right_);
	SAFE_DELETE(cost_init_);
	SAFE_DELETE(cost_aggr_);
	SAFE_DELETE(cost_aggr_1_);
	SAFE_DELETE(cost_aggr_2_);
	SAFE_DELETE(cost_aggr_3_);
	SAFE_DELETE(cost_aggr_4_);
	SAFE_DELETE(cost_aggr_5_);
	SAFE_DELETE(cost_aggr_6_);
	SAFE_DELETE(cost_aggr_7_);
	SAFE_DELETE(cost_aggr_8_);
	SAFE_DELETE(disp_left_);
	SAFE_DELETE(disp_right_);
}

bool SemiGlobalMatching::Match(const uint8_t* img_left, const uint8_t* img_right, float* disp_left)
{
	if (!is_initialized_) {
		return false;
	}
	if (img_left == nullptr || img_right == nullptr) {
		return false;
	}

	img_left_ = img_left;
	img_right_ = img_right;


	CensusTransform();
	ComputeCost();
	CostAggregation();
	ComputeDisparity();


	if (option_.is_check_lr) {
		ComputeDisparityRight();
		LRCheck();
	}

	if (option_.is_remove_speckles) {
		RemoveSpeckles(disp_left_, width_, height_, 2.0f, option_.min_speckle_aera, INVALID_FLOAT);
	}

	if (option_.is_fill_holes) {
		FillHolesInDispMap();
	}

	MedianFilter(disp_left_, disp_left_, width_, height_, 3);
	memcpy(disp_left, disp_left_, height_ * width_ * sizeof(float));

	return true;
}

bool SemiGlobalMatching::Reset(const uint32_t& width, const uint32_t& height, const SGMOption& option)
{

	Release();
	is_initialized_ = false;
	return Initialize(width, height, option);
}

void SemiGlobalMatching::census_transform_5x5(const uint8_t* source, uint32_t* census, const int32_t& width,
	const int32_t& height)
{
	if (source == nullptr || census == nullptr || width <= 5u || height <= 5u) {
		return;
	}


	for (int32_t i = 2; i < height - 2; i++) {
		for (int32_t j = 2; j < width - 2; j++) {
			const uint8_t gray_center = source[i * width + j];
			uint32_t census_val = 0u;
			for (int32_t r = -2; r <= 2; r++) {
				for (int32_t c = -2; c <= 2; c++) {
					census_val <<= 1;
					const uint8_t gray = source[(i + r) * width + j + c];
					if (gray < gray_center) {
						census_val += 1;
					}
				}
			}

			census[i * width + j] = census_val;
		}
	}
}

void SemiGlobalMatching::CensusTransform()
{
	census_transform_5x5(img_left_, census_left_, width_, height_);
	census_transform_5x5(img_right_, census_right_, width_, height_);
}

void SemiGlobalMatching::ComputeCost()
{
	const int32_t& min_disparity = option_.min_disparity;
	const int32_t& max_disparity = option_.max_disparity;
	const int32_t disp_range = max_disparity - min_disparity;
	if (disp_range <= 0) {
		return;
	}


	for (int32_t i = 0; i < height_; i++) {
		for (int32_t j = 0; j < width_; j++) {

			const uint32_t census_val_l = census_left_[i * width_ + j];

			for (int32_t d = min_disparity; d < max_disparity; d++) {
				auto& cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
				if (j - d < 0 || j - d >= width_) {
					cost = UINT8_MAX;
					continue;
				}
				const uint32_t census_val_r = census_right_[i * width_ + j - d];

				cost = Hamming32(census_val_l, census_val_r);
			}
		}
	}
}

void SemiGlobalMatching::CostAggregateLeftRight(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward) {

	const int32_t disp_range = max_disparity - min_disparity;

	const auto& P1 = p1;
	const auto& P2_Init = p2_init;

	const int32_t direction = is_forward ? 1 : -1;

	for (int32_t i = 0u; i < height; i++) {

		auto cost_init_row = (is_forward) ? (cost_init + i * width * disp_range) : (cost_init + i * width * disp_range + (width - 1) * disp_range);
		auto cost_aggr_row = (is_forward) ? (cost_aggr + i * width * disp_range) : (cost_aggr + i * width * disp_range + (width - 1) * disp_range);
		auto img_row = (is_forward) ? (img_data + i * width) : (img_data + i * width + width - 1);


		uint8_t gray = *img_row;
		uint8_t gray_last = *img_row;

		std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

		memcpy(cost_aggr_row, cost_init_row, disp_range * sizeof(uint8_t));
		memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8_t));
		cost_init_row += direction * disp_range;
		cost_aggr_row += direction * disp_range;
		img_row += direction;

		uint8_t mincost_last_path = UINT8_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}

		for (int32_t j = 0; j < width - 1; j++) {
			gray = *img_row;
			uint8_t min_cost = UINT8_MAX;
			for (int32_t d = 0; d < disp_range; d++) {
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8_t  cost = cost_init_row[d];
				const uint16_t l1 = cost_last_path[d + 1];
				const uint16_t l2 = cost_last_path[d] + P1;
				const uint16_t l3 = cost_last_path[d + 2] + P1;
				const uint16_t l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));

				const uint8_t cost_s = cost + static_cast<uint8_t>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);

				cost_aggr_row[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8_t));

			cost_init_row += direction * disp_range;
			cost_aggr_row += direction * disp_range;
			img_row += direction;

			gray_last = gray;
		}
	}
}

void SemiGlobalMatching::CostAggregateUpDown(const uint8_t* img_data, const int32_t& width, const int32_t& height,
	const int32_t& min_disparity, const int32_t& max_disparity, const int32_t& p1, const int32_t& p2_init,
	const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward)
{

	const int32_t disp_range = max_disparity - min_disparity;

	const auto& P1 = p1;
	const auto& P2_Init = p2_init;


	const int32_t direction = is_forward ? 1 : -1;

	for (int32_t j = 0; j < width; j++) {

		auto cost_init_col = (is_forward) ? (cost_init + j * disp_range) : (cost_init + (height - 1) * width * disp_range + j * disp_range);
		auto cost_aggr_col = (is_forward) ? (cost_aggr + j * disp_range) : (cost_aggr + (height - 1) * width * disp_range + j * disp_range);
		auto img_col = (is_forward) ? (img_data + j) : (img_data + (height - 1) * width + j);

		uint8_t gray = *img_col;
		uint8_t gray_last = *img_col;


		std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);


		memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
		memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));
		cost_init_col += direction * width * disp_range;
		cost_aggr_col += direction * width * disp_range;
		img_col += direction * width;

		uint8_t mincost_last_path = UINT8_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}


		for (int32_t i = 0; i < height - 1; i++) {
			gray = *img_col;
			uint8_t min_cost = UINT8_MAX;
			for (int32_t d = 0; d < disp_range; d++) {
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8_t  cost = cost_init_col[d];
				const uint16_t l1 = cost_last_path[d + 1];
				const uint16_t l2 = cost_last_path[d] + P1;
				const uint16_t l3 = cost_last_path[d + 2] + P1;
				const uint16_t l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));

				const uint8_t cost_s = cost + static_cast<uint8_t>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);

				cost_aggr_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

			cost_init_col += direction * width * disp_range;
			cost_aggr_col += direction * width * disp_range;
			img_col += direction * width;

			gray_last = gray;
		}
	}
}

void SemiGlobalMatching::CostAggregateDagonal_1(const uint8_t* img_data, const int32_t& width, const int32_t& height,
	const int32_t& min_disparity, const int32_t& max_disparity, const int32_t& p1, const int32_t& p2_init,
	const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward)
{
	const int32_t disp_range = max_disparity - min_disparity;

	const auto& P1 = p1;
	const auto& P2_Init = p2_init;

	const int32_t direction = is_forward ? 1 : -1;

	int32_t current_row = 0;
	int32_t current_col = 0;

	for (int32_t j = 0; j < width; j++) {

		auto cost_init_col = (is_forward) ? (cost_init + j * disp_range) : (cost_init + (height - 1) * width * disp_range + j * disp_range);
		auto cost_aggr_col = (is_forward) ? (cost_aggr + j * disp_range) : (cost_aggr + (height - 1) * width * disp_range + j * disp_range);
		auto img_col = (is_forward) ? (img_data + j) : (img_data + (height - 1) * width + j);

		std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);


		memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
		memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

		uint8_t gray = *img_col;
		uint8_t gray_last = *img_col;


		current_row = is_forward ? 0 : height - 1;
		current_col = j;
		if (is_forward && current_col == width - 1 && current_row < height - 1) {
			cost_init_col = cost_init + (current_row + direction) * width * disp_range;
			cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
			img_col = img_data + (current_row + direction) * width;
		}
		else if (!is_forward && current_col == 0 && current_row > 0) {
			cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
			cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
			img_col = img_data + (current_row + direction) * width + (width - 1);
		}
		else {
			cost_init_col += direction * (width + 1) * disp_range;
			cost_aggr_col += direction * (width + 1) * disp_range;
			img_col += direction * (width + 1);
		}

		uint8_t mincost_last_path = UINT8_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}

		for (int32_t i = 0; i < height - 1; i++) {
			gray = *img_col;
			uint8_t min_cost = UINT8_MAX;
			for (int32_t d = 0; d < disp_range; d++) {
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8_t  cost = cost_init_col[d];
				const uint16_t l1 = cost_last_path[d + 1];
				const uint16_t l2 = cost_last_path[d] + P1;
				const uint16_t l3 = cost_last_path[d + 2] + P1;
				const uint16_t l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));

				const uint8_t cost_s = cost + static_cast<uint8_t>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);

				cost_aggr_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));


			current_row += direction;
			current_col += direction;


			if (is_forward && current_col == width - 1 && current_row < height - 1) {
				cost_init_col = cost_init + (current_row + direction) * width * disp_range;
				cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
				img_col = img_data + (current_row + direction) * width;
			}
			else if (!is_forward && current_col == 0 && current_row > 0) {
				cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
				cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
				img_col = img_data + (current_row + direction) * width + (width - 1);
			}
			else {
				cost_init_col += direction * (width + 1) * disp_range;
				cost_aggr_col += direction * (width + 1) * disp_range;
				img_col += direction * (width + 1);
			}

			gray_last = gray;
		}
	}
}

void SemiGlobalMatching::CostAggregateDagonal_2(const uint8_t* img_data, const int32_t& width, const int32_t& height,
	const int32_t& min_disparity, const int32_t& max_disparity, const int32_t& p1, const int32_t& p2_init,
	const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward)
{
	const int32_t disp_range = max_disparity - min_disparity;


	const auto& P1 = p1;
	const auto& P2_Init = p2_init;


	const int32_t direction = is_forward ? 1 : -1;


	int32_t current_row = 0;
	int32_t current_col = 0;

	for (int32_t j = 0; j < width; j++) {
		auto cost_init_col = (is_forward) ? (cost_init + j * disp_range) : (cost_init + (height - 1) * width * disp_range + j * disp_range);
		auto cost_aggr_col = (is_forward) ? (cost_aggr + j * disp_range) : (cost_aggr + (height - 1) * width * disp_range + j * disp_range);
		auto img_col = (is_forward) ? (img_data + j) : (img_data + (height - 1) * width + j);

		std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

		memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
		memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

		uint8_t gray = *img_col;
		uint8_t gray_last = *img_col;

		current_row = is_forward ? 0 : height - 1;
		current_col = j;
		if (is_forward && current_col == 0 && current_row < height - 1) {
			cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
			cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
			img_col = img_data + (current_row + direction) * width + (width - 1);
		}
		else if (!is_forward && current_col == width - 1 && current_row > 0) {
			cost_init_col = cost_init + (current_row + direction) * width * disp_range;
			cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
			img_col = img_data + (current_row + direction) * width;
		}
		else {
			cost_init_col += direction * (width - 1) * disp_range;
			cost_aggr_col += direction * (width - 1) * disp_range;
			img_col += direction * (width - 1);
		}

		uint8_t mincost_last_path = UINT8_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}

		for (int32_t i = 0; i < height - 1; i++) {
			gray = *img_col;
			uint8_t min_cost = UINT8_MAX;
			for (int32_t d = 0; d < disp_range; d++) {
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8_t  cost = cost_init_col[d];
				const uint16_t l1 = cost_last_path[d + 1];
				const uint16_t l2 = cost_last_path[d] + P1;
				const uint16_t l3 = cost_last_path[d + 2] + P1;
				const uint16_t l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));

				const uint8_t cost_s = cost + static_cast<uint8_t>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);

				cost_aggr_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

			current_row += direction;
			current_col -= direction;

			if (is_forward && current_col == 0 && current_row < height - 1) {
				cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
				cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
				img_col = img_data + (current_row + direction) * width + (width - 1);
			}
			else if (!is_forward && current_col == width - 1 && current_row > 0) {
				cost_init_col = cost_init + (current_row + direction) * width * disp_range;
				cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
				img_col = img_data + (current_row + direction) * width;
			}
			else {
				cost_init_col += direction * (width - 1) * disp_range;
				cost_aggr_col += direction * (width - 1) * disp_range;
				img_col += direction * (width - 1);
			}

			gray_last = gray;
		}
	}
}

void SemiGlobalMatching::MedianFilter(const float* in, float* out, const int32_t& width, const int32_t& height,
	const int32_t wnd_size)
{
	const int32_t radius = wnd_size / 2;
	const int32_t size = wnd_size * wnd_size;

	std::vector<float> wnd_data;
	wnd_data.reserve(size);

	for (int32_t i = 0; i < height; i++) {
		for (int32_t j = 0; j < width; j++) {
			wnd_data.clear();

			for (int32_t r = -radius; r <= radius; r++) {
				for (int32_t c = -radius; c <= radius; c++) {
					const int32_t row = i + r;
					const int32_t col = j + c;
					if (row >= 0 && row < height && col >= 0 && col < width) {
						wnd_data.push_back(in[row * width + col]);
					}
				}
			}

			std::sort(wnd_data.begin(), wnd_data.end());
			out[i * width + j] = wnd_data[wnd_data.size() / 2];
		}
	}
}

void SemiGlobalMatching::RemoveSpeckles(float* disparity_map, const int32_t& width, const int32_t& height,
	const int32_t& diff_insame, const uint32_t& min_speckle_aera, const float& invalid_val)
{
	assert(width > 0 && height > 0);
	if (width < 0 || height < 0) {
		return;
	}

	std::vector<bool> visited(uint32_t(width * height), false);
	for (int32_t i = 0; i < height; i++) {
		for (int32_t j = 0; j < width; j++) {
			if (visited[i * width + j] || disparity_map[i * width + j] == invalid_val) {
				continue;
			}

			std::vector<std::pair<int32_t, int32_t>> vec;
			vec.emplace_back(i, j);
			visited[i * width + j] = true;
			uint32_t cur = 0;
			uint32_t next = 0;
			do {
				next = vec.size();
				for (uint32_t k = cur; k < next; k++) {
					const auto& pixel = vec[k];
					const int32_t row = pixel.first;
					const int32_t col = pixel.second;
					const auto& disp_base = disparity_map[row * width + col];
					for (int r = -1; r <= 1; r++) {
						for (int c = -1; c <= 1; c++) {
							if (r == 0 && c == 0) {
								continue;
							}
							int rowr = row + r;
							int colc = col + c;
							if (rowr >= 0 && rowr < height && colc >= 0 && colc < width) {
								if (!visited[rowr * width + colc] && abs(disparity_map[rowr * width + colc] - disp_base) <= diff_insame) {
									vec.emplace_back(rowr, colc);
									visited[rowr * width + colc] = true;
								}
							}
						}
					}
				}
				cur = next;
			} while (next < vec.size());

			if (vec.size() < min_speckle_aera) {
				for (auto& pix : vec) {
					disparity_map[pix.first * width + pix.second] = invalid_val;

				}
			}
		}
	}
}

void SemiGlobalMatching::CostAggregation() 
{

	const auto& min_disparity = option_.min_disparity;
	const auto& max_disparity = option_.max_disparity;
	assert(max_disparity > min_disparity);

	const int32_t size = width_ * height_ * (max_disparity - min_disparity);
	if (size <= 0) {
		return;
	}

	const auto& P1 = option_.p1;
	const auto& P2_Int = option_.p2_init;

	if (option_.num_paths == 4 || option_.num_paths == 8) {

		CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_1_, true);
		CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_2_, false);

		CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_3_, true);
		CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_4_, false);
	}

	if (option_.num_paths == 8) {

		CostAggregateDagonal_1(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_5_, true);
		CostAggregateDagonal_1(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_6_, false);

		CostAggregateDagonal_2(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_7_, true);
		CostAggregateDagonal_2(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_8_, false);
	}


	for (int32_t i = 0; i < size; i++) {
		if (option_.num_paths == 4 || option_.num_paths == 8) {
			cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
		}
		if (option_.num_paths == 8) {
			cost_aggr_[i] += cost_aggr_5_[i] + cost_aggr_6_[i] + cost_aggr_7_[i] + cost_aggr_8_[i];
		}
	}
}

void SemiGlobalMatching::ComputeDisparity()
{
	const int32_t& min_disparity = option_.min_disparity;
	const int32_t& max_disparity = option_.max_disparity;
	const int32_t disp_range = max_disparity - min_disparity;
	if (disp_range <= 0) {
		return;
	}

	// 左影像视差图
	const auto disparity = disp_left_;
	// 左影像聚合代价数组
	const auto cost_ptr = cost_aggr_;

	const int32_t width = width_;
	const int32_t height = height_;
	const bool is_check_unique = option_.is_check_unique;
	const float uniqueness_ratio = option_.uniqueness_ratio;

	// 为了加快读取效率，把单个像素的所有代价值存储到局部数组里
	std::vector<uint16_t> cost_local(disp_range);

	// ---逐像素计算最优视差
	for (int32_t i = 0; i < height; i++) {
		for (int32_t j = 0; j < width; j++) {
			uint16_t min_cost = UINT16_MAX;
			uint16_t sec_min_cost = UINT16_MAX;
			int32_t best_disparity = 0;

			// ---遍历视差范围内的所有代价值，输出最小代价值及对应的视差值
			for (int32_t d = min_disparity; d < max_disparity; d++) {
				const int32_t d_idx = d - min_disparity;
				const auto& cost = cost_local[d_idx] = cost_ptr[i * width * disp_range + j * disp_range + d_idx];
				if (min_cost > cost) {
					min_cost = cost;
					best_disparity = d;
				}
			}

			if (is_check_unique) {
				// 再遍历一次，输出次最小代价值
				for (int32_t d = min_disparity; d < max_disparity; d++) {
					if (d == best_disparity) {
						// 跳过最小代价值
						continue;
					}
					const auto& cost = cost_local[d - min_disparity];
					sec_min_cost = std::min(sec_min_cost, cost);
				}

				// 判断唯一性约束
				// 若(min-sec)/min < min*(1-uniquness)，则为无效估计
				if (sec_min_cost - min_cost <= static_cast<uint16_t>(min_cost * (1 - uniqueness_ratio))) {
					disparity[i * width + j] = INVALID_FLOAT;
					continue;
				}
			}

			// ---子像素拟合
			if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
				disparity[i * width + j] = INVALID_FLOAT;
				continue;
			}
			// 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
			const int32_t idx_1 = best_disparity - 1 - min_disparity;
			const int32_t idx_2 = best_disparity + 1 - min_disparity;
			const uint16_t cost_1 = cost_local[idx_1];
			const uint16_t cost_2 = cost_local[idx_2];
			// 解一元二次曲线极值
			const uint16_t denom = std::max(1, cost_1 + cost_2 - 2 * min_cost);
			disparity[i * width + j] = static_cast<float>(best_disparity) + static_cast<float>(cost_1 - cost_2) / (denom * 2.0f);
		}
	}
}

void SemiGlobalMatching::ComputeDisparityRight()
{
	const int32_t& min_disparity = option_.min_disparity;
	const int32_t& max_disparity = option_.max_disparity;
	const int32_t disp_range = max_disparity - min_disparity;
	if (disp_range <= 0) {
		return;
	}

	// 右影像视差图
	const auto disparity = disp_right_;
	// 左影像聚合代价数组
	const auto cost_ptr = cost_aggr_;

	const int32_t width = width_;
	const int32_t height = height_;
	const bool is_check_unique = option_.is_check_unique;
	const float uniqueness_ratio = option_.uniqueness_ratio;

	// 为了加快读取效率，把单个像素的所有代价值存储到局部数组里
	std::vector<uint16_t> cost_local(disp_range);

	// ---逐像素计算最优视差
	// 通过左影像的代价，获取右影像的代价
	// 右cost(xr,yr,d) = 左cost(xr+d,yl,d)
	for (int32_t i = 0; i < height; i++) {
		for (int32_t j = 0; j < width; j++) {
			uint16_t min_cost = UINT16_MAX;
			uint16_t sec_min_cost = UINT16_MAX;
			int32_t best_disparity = 0;

			// ---统计候选视差下的代价值
			for (int32_t d = min_disparity; d < max_disparity; d++) {
				const int32_t d_idx = d - min_disparity;
				const int32_t col_left = j + d;
				if (col_left >= 0 && col_left < width) {
					const auto& cost = cost_local[d_idx] = cost_ptr[i * width * disp_range + col_left * disp_range + d_idx];
					if (min_cost > cost) {
						min_cost = cost;
						best_disparity = d;
					}
				}
				else {
					cost_local[d_idx] = UINT16_MAX;
				}
			}

			if (is_check_unique) {
				// 再遍历一次，输出次最小代价值
				for (int32_t d = min_disparity; d < max_disparity; d++) {
					if (d == best_disparity) {
						// 跳过最小代价值
						continue;
					}
					const auto& cost = cost_local[d - min_disparity];
					sec_min_cost = std::min(sec_min_cost, cost);
				}

				// 判断唯一性约束
				// 若(min-sec)/min < min*(1-uniquness)，则为无效估计
				if (sec_min_cost - min_cost <= static_cast<uint16_t>(min_cost * (1 - uniqueness_ratio))) {
					disparity[i * width + j] = INVALID_FLOAT;
					continue;
				}
			}

			// ---子像素拟合
			if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
				disparity[i * width + j] = INVALID_FLOAT;
				continue;
			}

			// 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
			const int32_t idx_1 = best_disparity - 1 - min_disparity;
			const int32_t idx_2 = best_disparity + 1 - min_disparity;
			const uint16_t cost_1 = cost_local[idx_1];
			const uint16_t cost_2 = cost_local[idx_2];
			// 解一元二次曲线极值
			const uint16_t denom = std::max(1, cost_1 + cost_2 - 2 * min_cost);
			disparity[i * width + j] = static_cast<float>(best_disparity) + static_cast<float>(cost_1 - cost_2) / (denom * 2.0f);
		}
	}
}

void SemiGlobalMatching::LRCheck()
{
	const int width = width_;
	const int height = height_;

	const float& threshold = option_.lrcheck_thres;

	// 遮挡区像素和误匹配区像素
	auto& occlusions = occlusions_;
	auto& mismatches = mismatches_;
	occlusions.clear();
	mismatches.clear();

	// ---左右一致性检查
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			// 左影像视差值
			auto& disp = disp_left_[i * width + j];

			if (disp == INVALID_FLOAT) {
				mismatches.emplace_back(i, j);
				continue;
			}

			// 根据视差值找到右影像上对应的同名像素
			const auto col_right = static_cast<int32_t>(j - disp + 0.5);

			if (col_right >= 0 && col_right < width) {

				// 右影像上同名像素的视差值
				const auto& disp_r = disp_right_[i * width + col_right];

				// 判断两个视差值是否一致（差值在阈值内）
				if (abs(disp - disp_r) > threshold) {
					// 区分遮挡区和误匹配区
					// 通过右影像视差算出在左影像的匹配像素，并获取视差disp_rl
					// if(disp_rl > disp) 
					//		pixel in occlusions
					// else 
					//		pixel in mismatches
					const int32_t col_rl = static_cast<int32_t>(col_right + disp_r + 0.5);
					if (col_rl > 0 && col_rl < width) {
						const auto& disp_l = disp_left_[i * width + col_rl];
						if (disp_l > disp) {
							occlusions.emplace_back(i, j);
						}
						else {
							mismatches.emplace_back(i, j);
						}
					}
					else {
						mismatches.emplace_back(i, j);
					}

					// 让视差值无效
					disp = INVALID_FLOAT;
				}
			}
			else {
				// 通过视差值在右影像上找不到同名像素（超出影像范围）
				disp = INVALID_FLOAT;
				mismatches.emplace_back(i, j);
			}
		}
	}
}

void SemiGlobalMatching::FillHolesInDispMap()
{
	const int32_t width = width_;
	const int32_t height = height_;

	std::vector<float> disp_collects;

	// 定义8个方向
	float pi = 3.1415926;
	float angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
	float angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
	float* angle = angle1;

	float* disp_ptr = disp_left_;
	for (int k = 0; k < 3; k++) {
		// 第一次循环处理遮挡区，第二次循环处理误匹配区
		auto& trg_pixels = (k == 0) ? occlusions_ : mismatches_;

		std::vector<std::pair<int, int>> inv_pixels;
		if (k == 2) {
			//  第三次循环处理前两次没有处理干净的像素
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (disp_ptr[i * width + j] == INVALID_FLOAT) {
						inv_pixels.emplace_back(i, j);
					}
				}
			}
			trg_pixels = inv_pixels;
		}

		// 遍历待处理像素
		for (auto& pix : trg_pixels) {
			int y = pix.first;
			int x = pix.second;

			if (y == height / 2) {
				angle = angle2;
			}

			// 收集8个方向上遇到的首个有效视差值
			disp_collects.clear();
			for (int32_t n = 0; n < 8; n++) {
				const float ang = angle[n];
				const float sina = sin(ang);
				const float cosa = cos(ang);
				for (int32_t n = 1; ; n++) {
					const int32_t yy = y + n * sina;
					const int32_t xx = x + n * cosa;
					if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
						break;
					}
					auto& disp = *(disp_ptr + yy * width + xx);
					if (disp != INVALID_FLOAT) {
						disp_collects.push_back(disp);
						break;
					}
				}
			}
			if (disp_collects.empty()) {
				continue;
			}

			std::sort(disp_collects.begin(), disp_collects.end());

			// 如果是遮挡区，则选择第二小的视差值
			// 如果是误匹配区，则选择中值
			if (k == 0) {
				if (disp_collects.size() > 1) {
					disp_ptr[y * width + x] = disp_collects[1];
				}
				else {
					disp_ptr[y * width + x] = disp_collects[0];
				}
			}
			else {
				disp_ptr[y * width + x] = disp_collects[disp_collects.size() / 2];
			}
		}
	}
}