#pragma once

#include <QObject>

#include <opencv2/core.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"

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


class Live_Stitcher : public QObject
{
	Q_OBJECT

public:
	Live_Stitcher( Camera_Interface* camera, QObject *parent);
	~Live_Stitcher();

	void Start_Thread();

signals:
	void Display_Image( const cv::Mat & image );
	void Work_Finished();

private:
	void Stitch_Loop();
	void GPU_Stitch_Loop();
	cv::Mat current_image;
	cv::Mat Overall_Image;
	cv::Ptr<cv::ORB> detector;
	cv::Ptr<cv::BFMatcher> matcher;
	cv::Mat all_descriptors;
	std::vector<cv::KeyPoint> all_keypoints;

	cv::Ptr<cv::cuda::ORB> gpu_detector;
	cv::Ptr< cv::cuda::DescriptorMatcher > gpu_matcher;
	cv::cuda::GpuMat gpu_all_descriptors;

	cv::Ptr<cv::superres::SuperResolution> superRes;
	cv::Ptr<FramePassthrough> frame_passthrough;

	Camera_Interface* camera;
};
