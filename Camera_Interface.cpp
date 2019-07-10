#include "Camera_Interface.h"

#include <QTimer>
#include "opencv2/imgcodecs.hpp"

using namespace cv;

Camera_Interface::Camera_Interface(QObject *parent)
	: QObject(parent)
{
}

Camera_Interface::~Camera_Interface()
{
}

void Camera_Interface::Start_Thread()
{
	Start_Camera( 1920, 1080 );

	// Create timer to continuously run loop, but run other events between calls
	QTimer* camera_loop_timer = new QTimer( this );
	connect( camera_loop_timer, &QTimer::timeout, this, &Camera_Interface::Read_Camera_Loop );
	camera_loop_timer->start( 1000 / 60 );
}

void Camera_Interface::Start_Camera( int resolution_x, int resolution_y )
{
	//capture_device.open( 0 ); // open the default camera
	//capture_device.open( 1 ); // open the default camera
	//capture_device.open( 1, cv::CAP_DSHOW ); // open the default camera
	capture_device.open( 0, cv::CAP_MSMF ); // open the default camera
	//capture_device.set( CAP_PROP_FOURCC, cv::VideoWriter::fourcc( 'm', 'j', 'p', '2' ) );
	//capture_device.set( CAP_PROP_FOURCC, cv::VideoWriter::fourcc( 'M', 'J', 'P', 'G' ) );
	//const int MODE_HW = 1;
	//const int CV_CAP_PROP_MODE = 9;
	//capture_device.set( CV_CAP_PROP_MODE, MODE_HW );
	capture_device.set( CAP_PROP_FPS, 60 );
	//capture_device.set( CAP_PROP_FORMAT, CV_8UC3 );
	//capture_device.set( CAP_PROP_FOURCC, cv::VideoWriter::fourcc( 'H', '2', '6', '4' ) );
	//capture_device.set( CAP_PROP_FRAME_WIDTH, 640 );
	//capture_device.set( CAP_PROP_FRAME_HEIGHT, 480 );
	//capture_device.set( CAP_PROP_FRAME_WIDTH, 1280 );
	//capture_device.set( CAP_PROP_FRAME_HEIGHT, 720 );
	capture_device.set( CAP_PROP_FRAME_WIDTH, resolution_x );
	capture_device.set( CAP_PROP_FRAME_HEIGHT, resolution_y );
	//capture_device.set( CAP_PROP_CONVERT_RGB, 0 );
	//capture_device.set( CAP_PROP_FRAME_WIDTH, 3840 );
	//capture_device.set( CAP_PROP_FRAME_HEIGHT, 2160 );
	//capture_device.set( CAP_PROP_FRAME_WIDTH, 4224 );
	//capture_device.set( CAP_PROP_FRAME_HEIGHT, 3156 );
}

void Camera_Interface::Read_Camera_Loop()
{
	std::lock_guard<std::mutex> guard( image_mutex[ image_index ] );
	try
	{
		capture_device >> current_image[ image_index ];
	}
	catch( ... )
	{
		QThread::msleep( 100 );
		Start_Camera( 1920, 1080 );
	}
	image_index = (image_index + 1) % NUMBER_OF_BUFFER_IMAGES;
	//static int test_index = 0;
	//if( test_index > 30 )
	//{
	//	capture_device.set( CAP_PROP_FRAME_WIDTH, 640 );
	//	capture_device.set( CAP_PROP_FRAME_HEIGHT, 480 );
	//}
	//else
	//{
	//	capture_device.set( CAP_PROP_FRAME_WIDTH, 1920 );
	//	capture_device.set( CAP_PROP_FRAME_HEIGHT, 1080 );
	//}
	//if( test_index > 120 )
	//	test_index = 0;
	//else
	//	test_index++;
	//QThread::msleep( 1000 );
}

void Camera_Interface::Take_Image()
{
	QString file_name = "test.jpg";
	std::lock_guard<std::mutex> guard( image_mutex[ image_index ] );
	printf( "start" );
	bool success = false;
	while( success == false )
	{
		try
		{
			capture_device.open( 1, cv::CAP_DSHOW ); // open the default camera
			QThread::msleep( 10 );
			Start_Camera( 4224, 3156 );
			capture_device >> current_image[ image_index ];
			QThread::msleep( 10 );
			capture_device.open( 1, cv::CAP_DSHOW ); // open the default camera
			QThread::msleep( 10 );
			Start_Camera( 4224, 3156 );
			//Start_Camera( 1920, 1080 );
			cv::imwrite( file_name.toStdString(), current_image[ image_index ] );
			success = true;
		}
		catch(...)
		{
			QThread::msleep( 100 );
		}
	}
	image_index = (image_index + 1) % 2;
	printf( "end" );
}

Mat Camera_Interface::Get_Image()
{
	//Mat result;
	//QMetaObject::invokeMethod( this, [ this ]
	//{
	//	return this->current_image;
	//}, Qt::BlockingQueuedConnection, &result );

	int last_finished_image_index = (image_index + NUMBER_OF_BUFFER_IMAGES - 1) % NUMBER_OF_BUFFER_IMAGES;
	std::lock_guard<std::mutex> guard( image_mutex[ last_finished_image_index ] );
	return this->current_image[ last_finished_image_index ];
}