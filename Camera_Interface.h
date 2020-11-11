#pragma once

#include <QObject>
#include <QThread>
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "Pleasant_OpenCV.h"

#include <mutex>

#define DEBUGGING_ON_LAPPYTOP

class Camera_Interface : public QObject
{
	Q_OBJECT

public:
	Camera_Interface(QObject *parent = nullptr);
	~Camera_Interface();

	void Start_Thread();
	void Read_Camera_Loop();
	pcv::RGBA_UChar_Image Get_Image();
	void Take_Image();

signals:
	void Work_Finished();

private:
	void Start_Camera( int resolution_x, int resolution_y );
		
	// Bufferring 4 images so that reference counts stay at 1 to allow memory reuse on new frames
	// Smaller buffer might mean image is still being used somewhere (ref count > 2) when camera is ready to
	// write a new image, so it will need to allocate new memory
	static const int NUMBER_OF_BUFFER_IMAGES = 4;
	int image_index{ 0 };
	pcv::RGBA_UChar_Image current_image[ NUMBER_OF_BUFFER_IMAGES ]; // Double buffer
	std::mutex image_mutex[ NUMBER_OF_BUFFER_IMAGES ];
	cv::VideoCapture capture_device;

	//int default_x_resolution = 640;
	//int default_y_resolution = 480;
	//int default_x_resolution = 1280;
	//int default_y_resolution = 720;
	//int default_x_resolution = 1920;
	//int default_y_resolution = 1080;
	//int picture_x_resolution = 1920;
	//int picture_y_resolution = 1080;

#ifdef DEBUGGING_ON_LAPPYTOP
	//int default_x_resolution = 4224;
	//int default_y_resolution = 3156;
	//int picture_x_resolution = 4224;
	//int picture_y_resolution = 3156;
	int default_x_resolution = 1920;
	int default_y_resolution = 1080;
	int picture_x_resolution = 1920;
	int picture_y_resolution = 1080;
#else
	int default_x_resolution = 1920;
	int default_y_resolution = 1080;
	int picture_x_resolution = 1920;
	int picture_y_resolution = 1080;
	//int default_x_resolution = 3840;
	//int default_y_resolution = 2160;
	//int picture_x_resolution = 3840;
	//int picture_y_resolution = 2160;
#endif


	qint64 time_sum = 0;
	int sum_counter = 0;
};
