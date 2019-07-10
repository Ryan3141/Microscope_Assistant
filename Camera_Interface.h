#pragma once

#include <QObject>
#include <QThread>
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"

#include <mutex>

class Camera_Interface : public QObject
{
	Q_OBJECT

public:
	Camera_Interface(QObject *parent = nullptr);
	~Camera_Interface();

	void Start_Thread();
	void Read_Camera_Loop();
	cv::Mat Get_Image();
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
	cv::Mat current_image[ NUMBER_OF_BUFFER_IMAGES ]; // Double buffer
	std::mutex image_mutex[ NUMBER_OF_BUFFER_IMAGES ];
	cv::VideoCapture capture_device;
};
