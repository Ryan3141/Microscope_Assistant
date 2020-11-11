#include "Camera_Interface.h"

#include <QTimer>
#include <QDateTime>
#include <QDebug>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core.hpp>

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
	Start_Camera( default_x_resolution, default_y_resolution );

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
#ifdef DEBUGGING_ON_LAPPYTOP
	//capture_device.open( 0, cv::CAP_MSMF ); // open the default camera
	capture_device.open( 1, cv::CAP_MSMF ); // open the default camera
#else
	capture_device.open( 0, cv::CAP_MSMF ); // open the default camera
#endif
	//capture_device.open( 0, cv::CAP_DSHOW ); // open the default camera
	//capture_device.set( CAP_PROP_FOURCC, cv::VideoWriter::fourcc( 'm', 'j', 'p', '2' ) );
	//capture_device.set( CAP_PROP_FOURCC, cv::VideoWriter::fourcc( 'Y', 'U', 'Y', '2' ) );
	//capture_device.set( CAP_PROP_FOURCC, cv::VideoWriter::fourcc( 'M', 'J', 'P', 'G' ) );
	//const int MODE_HW = 1;
	//const int CV_CAP_PROP_MODE = 9;
	//capture_device.set( CV_CAP_PROP_MODE, MODE_HW );
	//capture_device.set( CAP_PROP_FPS, 60 );
	//capture_device.set( CAP_PROP_FORMAT, CV_8UC3 );
	//capture_device.set( CAP_PROP_FOURCC, cv::VideoWriter::fourcc( 'H', '2', '6', '4' ) );
	capture_device.set( CAP_PROP_FRAME_WIDTH, resolution_x );
	capture_device.set( CAP_PROP_FRAME_HEIGHT, resolution_y );
	//capture_device.set( CAP_PROP_CONVERT_RGB, 0 );
}

void Camera_Interface::Read_Camera_Loop()
{
	std::lock_guard<std::mutex> guard( image_mutex[ image_index ] );
	qint64 before_time = QDateTime::currentMSecsSinceEpoch();
	if( false )
	{
		static bool done_once = false;
		static QString path = R"(D:\School\Processing\Microscope_Computer\Microscope Images\Ryan\SED\PR For Etch\)";
		static QStringList file_names;
		if( !done_once )
		{
			for( int i = 1; i <= 25; i++ )
				file_names.push_back( QString( "476-4 (%1).jpg" ).arg( i ) );
			pcv::RGBA_UChar_Image loaded_img;
			pcv::Convert<COLOR_RGB2RGBA>( pcv::RGB_UChar_Image( cv::imread( (path + file_names[ 0 ]).toStdString() ) ), loaded_img );
			for( auto & image : current_image )
				image = loaded_img;
			done_once = true;
		}
		static int counter = 0;
		if( counter++ >= 10 )
		{
			int index = (counter / 10) % 21;
			pcv::RGBA_UChar_Image loaded_img;
			pcv::Convert<COLOR_RGB2RGBA>( pcv::RGB_UChar_Image( cv::imread( (path + file_names[ index ]).toStdString() ) ), loaded_img );
			for( auto & image : current_image )
				image = loaded_img;
		}
	}
	try
	{
		//QThread::msleep( 100 );
		Mat possible_new_frame;
		capture_device >> possible_new_frame;
		if( possible_new_frame.data != nullptr )
		{
			current_image[ image_index ] = possible_new_frame;
			sum_counter++;
		}
	}
	catch( ... )
	{
		QThread::msleep( 100 );
		Start_Camera( default_x_resolution, default_y_resolution );
	}
	image_index = (image_index + 1) % NUMBER_OF_BUFFER_IMAGES;
	qint64 after_time = QDateTime::currentMSecsSinceEpoch();
	time_sum += after_time - before_time;

	if( sum_counter >= 30 )
	{
		double fps = 1000.0 * sum_counter / time_sum;
		qDebug() << "Framerate from camera: " << fps;// << "\n";
		time_sum = 0;
		sum_counter = 0;
	}

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
	bool success = false;
	qint64 before_time = QDateTime::currentMSecsSinceEpoch();
	while( success == false )
	{
		try
		{
			//Start_Camera( picture_x_resolution, picture_x_resolution );
			//QThread::msleep( 10 );
			capture_device >> current_image[ image_index ];
			//Start_Camera( default_x_resolution, default_y_resolution );
			//QThread::msleep( 10 );
			//Start_Camera( 1920, 1080 );
			cv::imwrite( file_name.toStdString(), current_image[ image_index ] );
			success = true;
		}
		catch(...)
		{
			QThread::msleep( 100 );
			Start_Camera( default_x_resolution, default_y_resolution );
		}
	}
	image_index = (image_index + 1) % 2;
	qint64 after_time = QDateTime::currentMSecsSinceEpoch();
	qDebug() << "Picture took " << 1E-3 * (after_time - before_time) << "seconds\n";
}

pcv::RGBA_UChar_Image Camera_Interface::Get_Image()
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