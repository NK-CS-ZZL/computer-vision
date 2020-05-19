#pragma once
#include <cstdint>
#include <limits>
#include <vector>

#ifndef INVALID_FLOAT
#define INVALID_FLOAT std::numeric_limits<float>::infinity()
#endif


class SemiGlobalMatching
{
public:
	SemiGlobalMatching();
	~SemiGlobalMatching();


	struct SGMOption {
		uint8_t	num_paths;			
		int32_t  min_disparity;		
		int32_t	max_disparity;		

		bool	is_check_unique;	
		float	uniqueness_ratio;	

		bool	is_check_lr;		
		float	lrcheck_thres;		

		bool	is_remove_speckles;	
		int		min_speckle_aera;	

		bool	is_fill_holes;		


		int32_t  p1;				
		int32_t  p2_init;		

		SGMOption() : num_paths(8), min_disparity(0), max_disparity(640),
			is_check_unique(true), uniqueness_ratio(0.95f),
			is_check_lr(true), lrcheck_thres(1.0f),
			is_remove_speckles(true), min_speckle_aera(20),
			is_fill_holes(true),
			p1(10), p2_init(150)
		{
		}
	};
public:

	bool Initialize(const int32_t& width, const int32_t& height, const SGMOption& option);

	bool Match(const uint8_t* img_left, const uint8_t* img_right, float* disp_left);

	bool Reset(const uint32_t& width, const uint32_t& height, const SGMOption& option);

private:
	static inline uint8_t Hamming32(const uint32_t& x, const uint32_t& y)
	{
		uint32_t dist = 0, val = x ^ y;

		// Count the number of set bits
		while (val) {
			++dist;
			val &= val - 1;
		}

		return static_cast<uint8_t>(dist);
	}

	void CostAggregateLeftRight(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward);
	
	void CostAggregateUpDown(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	void CostAggregateDagonal_1(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	void CostAggregateDagonal_2(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	void census_transform_5x5(const uint8_t* source, uint32_t* census, const int32_t& width, const int32_t& height);

	void MedianFilter(const float* in, float* out, const int32_t& width, const int32_t& height, const int32_t wnd_size);

	void RemoveSpeckles(float* disparity_map, const int32_t& width, const int32_t& height, const int32_t& diff_insame, const uint32_t& min_speckle_aera, const float& invalid_val);

	void CensusTransform();

	void ComputeCost();

	void CostAggregation();

	void ComputeDisparity();

	void ComputeDisparityRight();

	void LRCheck();

	void FillHolesInDispMap();

	void Release();

private:

	SGMOption option_;


	int32_t width_;
	int32_t height_;

	const uint8_t* img_left_;
	const uint8_t* img_right_;


	uint32_t* census_left_;
	uint32_t* census_right_;


	uint8_t* cost_init_;


	uint16_t* cost_aggr_;

	uint8_t* cost_aggr_1_;
	uint8_t* cost_aggr_2_;
	uint8_t* cost_aggr_3_;
	uint8_t* cost_aggr_4_;
	uint8_t* cost_aggr_5_;
	uint8_t* cost_aggr_6_;
	uint8_t* cost_aggr_7_;
	uint8_t* cost_aggr_8_;

	float* disp_left_;
	float* disp_right_;

	bool is_initialized_;

	std::vector<std::pair<int, int>> occlusions_;
	std::vector<std::pair<int, int>> mismatches_;
};