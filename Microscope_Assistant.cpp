#include "Microscope_Assistant.h"

#include <QSettings>

#include "Device_Communicator.h"

//#pragma comment(lib, "opencv_world341.lib")
void test_main();

Microscope_Assistant::Microscope_Assistant( QWidget *parent )
	: QMainWindow(parent)
{
	test_main();
	settings = new QSettings( "configuration.ini", QSettings::IniFormat, this );

	Start_Looking_For_Connections( parent );
	ui.setupUi(this);
}

void Microscope_Assistant::Start_Looking_For_Connections( QWidget *parent )
{
	bool success = false;
	while( !success )
	{
		try
		{
			//QHostAddress listener_address( "192.168.1.198" );
			//QHostAddress listener_address( QHostAddress::LocalHost );
			QHostAddress listener_address;
			if( settings->contains( "StageServerInfo/Fixed_IP" ) )
				listener_address = QHostAddress( settings->value( "StageServerInfo/Fixed_IP" ).toString() );
			else
				listener_address = QHostAddress( QHostAddress::AnyIPv4 );
			unsigned short port = settings->value( "StageServerInfo/Listener_Port" ).toInt();
			my_devices = new Device_Communicator( parent, { { QString(), QString() } }, listener_address, port );
			my_devices->Poll_LocalIPs_For_Devices( settings->value( "SensorServerInfo/ip_range" ).toString() );
			qInfo() << QString( "Connected as %1 listening to port %2" ).arg( listener_address.toString() ).arg( port );
			success = true;
		}
		catch( const QString & error )
		{
			success = false;
			qCritical() << "Error with Device Communicator: " + error;
			int delay_seconds = 5;
			qCritical() << QString( "Trying again in %1 seconds" ).arg( delay_seconds );
			QThread::sleep( delay_seconds );
		}
	}
}
/*
* @file SURF_FlannMatcher
* @brief SURF detector + descriptor + FLANN Matcher
* @author A. Huaman
*/
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafeatures2d.hpp"

//#include "opencv2/stitching.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
/*
* @function main
* @brief Main function
*/
void test_main()
{
	VideoCapture cap( 0 ); // open the default camera
	cap.set( CAP_PROP_FRAME_WIDTH, 1920 );
	cap.set( CAP_PROP_FRAME_HEIGHT, 1080 );
	//VideoCapture cap( "Testing Video.mp4" );
	//VideoCapture cap( "D:\\Big Data\\test_normal_microscope1.mp4" );
	if( !cap.isOpened() )  // check if we succeeded
		return;
	Mat img1;
	while( 1 )
	{
		cap >> img1;
		cvtColor( img1, img1, cv::COLOR_BGR2GRAY );
		imshow( "Image 1", img1 );
		int i = 1 and 2;
		if( cv::waitKey( 10 ) == 's' )
		{
			break;
		}
	}

	//std::vector<cv::KeyPoint> kp_query;
	//cv::Mat descr_query;
	//cv::Ptr<cv::AKAZE> akaze2 = cv::AKAZE::create();
	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
	//Ptr<ORB> detector = ORB::create();
	//Ptr<SURF> detector = SURF::create();
	//int minHessian = 100; //400
	//detector->setHessianThreshold( minHessian );
	////-- Step 2: Matching descriptor vectors using FLANN matcher
	//FlannBasedMatcher matcher;
	//BFMatcher matcher( cv::NORM_HAMMING, true );

	cuda::GpuMat cuda_img1{ img1 };
	Ptr<cuda::ORB> detector = cuda::ORB::create();
	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

	while( 1 )
	{
		cv::Mat img2;
		cap >> img2;
		cvtColor( img2, img2, cv::COLOR_BGR2GRAY );
		cuda::GpuMat cuda_img2{ img2 };
		//cvtColor( img2, img2, CV_BGR2GRAY );

		if( !img1.data || !img2.data )
		{
			std::cout << " --(!) Error reading images " << std::endl;
			continue;
		}
		std::vector<KeyPoint> keypoints_1, keypoints_2;
		cuda::GpuMat descriptors_1, descriptors_2; // descriptors (features)
		detector->detectAndCompute( cuda_img1, cuda::GpuMat(), keypoints_1, descriptors_1 );
		detector->detectAndCompute( cuda_img2, cuda::GpuMat(), keypoints_2, descriptors_2 );
		//std::vector<KeyPoint> keypoints_1, keypoints_2;
		//Mat descriptors_1, descriptors_2;
		//detector->detectAndCompute( img1, Mat(), keypoints_1, descriptors_1 );
		//detector->detectAndCompute( img2, Mat(), keypoints_2, descriptors_2 );
		std::vector< DMatch > matches;
		if( descriptors_1.cols == 0 || descriptors_2.cols == 0 )
			continue;
		matcher->match( descriptors_1, descriptors_2, matches );
		double max_dist = 0; double min_dist = 100;
		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < matches.size(); i++ )
		{
			double dist = matches[ i ].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}
		printf( "-- Max dist : %f \n", max_dist );
		printf( "-- Min dist : %f \n", min_dist );
		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
		//-- or a small arbitrary value ( 0.02 ) in the event that min_dist is very
		//-- small)
		//-- PS.- radiusMatch can also be used here.
		std::vector< DMatch > good_matches;
		for( int i = 0; i < matches.size(); i++ )
		{
			if( matches[ i ].distance <= max( 1.2 * min_dist, 0.05 ) )
			{
				good_matches.push_back( matches[ i ] );
			}
		}
		//-- Draw only "good" matches
		Mat img_matches;
		drawMatches( img1, keypoints_1, img2, keypoints_2,
					 good_matches, img_matches, Scalar::all( -1 ), Scalar::all( -1 ),
					 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		{
			//-- Localize the object
			std::vector<Point2f> obj;
			std::vector<Point2f> scene;
			for( size_t i = 0; i < good_matches.size(); i++ )
			{
				//-- Get the keypoints from the good matches
				obj.push_back( keypoints_1[ good_matches[ i ].queryIdx ].pt );
				scene.push_back( keypoints_2[ good_matches[ i ].trainIdx ].pt );
				Point2f location_delta = keypoints_1[ good_matches[ i ].queryIdx ].pt - keypoints_2[ good_matches[ i ].trainIdx ].pt;
				double distance = location_delta.dot( location_delta );
			}
			Mat H = cv::findHomography( obj, scene, cv::RANSAC );
			if( H.data )
			{
				vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
				int solutions = cv::decomposeHomographyMat( H, cv::Mat::eye( H.size(), H.type() ), Rs_decomp, ts_decomp, normals_decomp );
				//int solutions = decomposeHomographyMat( homography, cameraMatrix, Rs_decomp, ts_decomp, normals_decomp );


				//-- Get the corners from the image_1 ( the object to be "detected" )
				std::vector<Point2f> obj_corners( 4 );
				obj_corners[ 0 ] = cv::Point( 0, 0 ); obj_corners[ 1 ] = cv::Point( img1.cols, 0 );
				obj_corners[ 2 ] = cv::Point( img1.cols, img1.rows ); obj_corners[ 3 ] = cv::Point( 0, img1.rows );
				std::vector<Point2f> scene_corners( 4 );
				perspectiveTransform( obj_corners, scene_corners, H );
				//-- Draw lines between the corners (the mapped object in the scene - image_2 )
				line( img_matches, scene_corners[ 0 ] + Point2f( img1.cols, 0 ), scene_corners[ 1 ] + Point2f( img1.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
				line( img_matches, scene_corners[ 1 ] + Point2f( img1.cols, 0 ), scene_corners[ 2 ] + Point2f( img1.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
				line( img_matches, scene_corners[ 2 ] + Point2f( img1.cols, 0 ), scene_corners[ 3 ] + Point2f( img1.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
				line( img_matches, scene_corners[ 3 ] + Point2f( img1.cols, 0 ), scene_corners[ 0 ] + Point2f( img1.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
			}
		}

		//-- Show detected matches
		imshow( "Good Matches", img_matches );
		imshow( "Image 1", img2 );
		for( int i = 0; i < (int)good_matches.size(); i++ )
		{
			printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[ i ].queryIdx, good_matches[ i ].trainIdx );
		}
		if( cv::waitKey( 10 ) == 's' )
		{
			break;
		}
	}
}
