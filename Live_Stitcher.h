#pragma once

#include <QObject>
#include <QFileInfo>

#include <opencv2/core.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"
#include "Pleasant_OpenCV.h"

class Camera_Interface;

class FramePassthrough : public cv::superres::FrameSource
{
public:
	cv::Mat frame_to_use;
	virtual void nextFrame( cv::OutputArray frame )
	{
		frame.getMatRef() = frame_to_use;
	}
	virtual void reset()
	{
	}
};

template<class Image_Type>
struct Image_Match_Info
{
	Image_Type img;
	Image_Type scaled_down_img;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	constexpr static int scale = 8;
};


class Live_Stitcher : public QObject
{
	Q_OBJECT

public:
	Live_Stitcher( Camera_Interface* camera, QObject *parent);
	~Live_Stitcher();

	void Start_Thread();
	void Stitch_Image_And_Start_New();
	void Save_Image_And_Start_Over( QFileInfo path_to_file );
	void Reset_Stitching();

signals:
	void Display_Image( pcv::BGRA_UChar_Image image );
	void Display_Debug_Image( pcv::BGRA_UChar_Image image );
	void Work_Finished();

private:
	void Stitch_Loop();
	void GPU_Stitch_Loop();

	void Find_Details_Mask( pcv::RGBA_UChar_Image & input, pcv::Gray_Float_Image & output ) const;

	pcv::RGBA_UChar_Image current_image;
	Image_Match_Info<pcv::RGBA_UChar_Image> Overall_Image;

	Image_Match_Info<pcv::RGBA_UChar_Image> previous_image;
	int cleaner_image_counter = 0;
	cv::Point previous_image_offset_in_overall_image = {};

	Image_Match_Info<pcv::RGBA_UChar_Image> proposed_image;
	//cv::Point proposed_shifted_origin = {};
	cv::Point proposed_offset_in_overall_image = {};
	cv::Mat alpha_mask;
	//cv::Mat Overall_Image;
	//cv::Mat all_descriptors;
	//std::vector<cv::KeyPoint> all_keypoints;

	cv::Ptr<cv::ORB> detector;
	cv::Ptr<cv::BFMatcher> matcher;

	cv::Ptr<cv::cuda::ORB> gpu_detector;
	cv::Ptr<cv::cuda::DescriptorMatcher> gpu_matcher;

	cv::Ptr<cv::superres::SuperResolution> superRes;
	cv::Ptr<FramePassthrough> frame_passthrough;

	Camera_Interface* camera;
};
