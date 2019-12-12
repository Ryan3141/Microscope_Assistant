#include "Live_Stitcher.h"

#include <QTimer>
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

#include "Pleasant_OpenCV.h"

#include "Camera_Interface.h"

//#include "opencv2/stitching.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

using namespace cv::superres;

Live_Stitcher::Live_Stitcher( Camera_Interface* camera, QObject *parent )
	: QObject( parent )
{
	this->camera = camera;
}

Live_Stitcher::~Live_Stitcher()
{
}

void Live_Stitcher::Start_Thread()
{
	// Create timer to continuously run loop, but run other events between calls
	this->detector = cv::ORB::create();
	this->matcher = cv::BFMatcher::create( cv::NORM_HAMMING, true );
	try
	{
		//this->gpu_detector = cuda::ORB::create();
		this->gpu_matcher = cuda::DescriptorMatcher::createBFMatcher( cv::NORM_HAMMING );
	}
	catch( ... )
	{
		printf( "Cuda not supported\n" );
	}
	this->frame_passthrough = Ptr<FramePassthrough>( new FramePassthrough() );

	const int scale = 2;// cmd.get<int>( "scale" );
	const int iterations = 4;// cmd.get<int>( "iterations" );
	const int temporalAreaRadius = 3;// cmd.get<int>( "temporal" );
	//this->superRes = createSuperResolution_BTVL1_CUDA();
	//superRes->setOpticalFlow( cv::superres::createOptFlow_DualTVL1_CUDA() );
	this->superRes = createSuperResolution_BTVL1();
	this->superRes->setOpticalFlow( cv::superres::createOptFlow_DualTVL1() );
	this->superRes->setScale( scale );
	this->superRes->setIterations( iterations );
	this->superRes->setTemporalAreaRadius( temporalAreaRadius );
	//this->superRes->setInput( createFrameSource_Camera( 0 ) );
	this->superRes->setInput( frame_passthrough );

	//Mat result;
	//for( int i = 0; i < 10; i++ )
	//	this->frame_passthrough->nextFrame( result );

	QTimer* loop_timer = new QTimer( this );
	connect( loop_timer, &QTimer::timeout, this, &Live_Stitcher::Stitch_Loop );
	loop_timer->start( 50 );
}

void Live_Stitcher::Stitch_Loop()
{
	this->current_image = camera->Get_Image();
	if( this->current_image.empty() )
		return;

	const pcv::RGB_UChar_Image & img = this->current_image;
	pcv::Gray_UChar_Image img_grayscale;
	//img_grayscale.create(img.size(), CV_8UC1);
	pcv::cvtColor<COLOR_RGB2GRAY>( img, img_grayscale );
	pcv::Gray_Float_Image grayscale;
	//grayscale.create( img.size(), CV_64F );
	pcv::Change_Data_Type( img_grayscale, grayscale );
	//cv::cvtColor( this->current_image, laplacian_img_bw, COLOR_BGR2GRAY );
	//cv::Laplacian( laplacian_img_bw, laplacian_img, CV_64F, 3 );
	pcv::Gray_Float_Image laplacian_img;
	//laplacian_img.create( img.size(), CV_64F );
	pcv::Laplacian( grayscale, laplacian_img, 7 );
	//pcv::multiply( laplacian_img, laplacian_img, laplacian_img, 1.0 );
	pcv::Gray_UChar_Image to_be_displayed_gray;
	//to_be_displayed_gray.create( img.size(), CV_8UC1 );
	pcv::Change_Data_Type( laplacian_img, to_be_displayed_gray );
	pcv::RGBA_UChar_Image to_be_displayed;
	//to_be_displayed.create( img.size(), CV_8UC4 );
	pcv::cvtColor<COLOR_GRAY2RGBA>( to_be_displayed_gray, to_be_displayed );
	emit Display_Image( to_be_displayed );
	return;
	constexpr int erosion_size = 1;
	Mat dilation_kernel = getStructuringElement( cv::MORPH_ELLIPSE,
										 Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ),
										 Point( erosion_size, erosion_size ) );
	cv::normalize( laplacian_img, laplacian_img, 0, 255.0, NORM_MINMAX );

	int threshold_value = 180;
	int max_BINARY_value = 0;
	//cv::threshold( laplacian_img, laplacian_img, threshold_value, max_BINARY_value, cv::THRESH_TOZERO );
	//cv::dilate( laplacian_img, laplacian_img, dilation_kernel );
	//cv::GaussianBlur( laplacian_img, laplacian_img, Size( erosion_size, erosion_size ), 5.0, 5.0 );


	//cv::equalizeHist( laplacian_img_bw, laplacian_img_bw ); // Only works on 8-bit data

	Mat laplacian_img_color;
	//cv::cvtColor( laplacian_img_bw, laplacian_img_color, COLOR_GRAY2BGR, CV_32SC4 );
	//imshow( "Test", laplacian_img );
	double* stupid_shit = (double*)laplacian_img.data + 8000;
	//laplacian_img.convertTo( laplacian_img, CV_8U );
	//for( int j = 0; j < 10; j++ )
	//{
	//	for( int i = 0; i < 10; i++ )
	//	{
	//		std::cout << int(laplacian_img.at<unsigned char>( i, j )) << " ";
	//	}
	//	std::cout << std::endl;
	//}
	unsigned char* fucking_thing = (unsigned char*)laplacian_img.data;
	Mat prepared_img( laplacian_img.size(), CV_8UC1 );// = laplacian_img.clone();
	Mat prepared_img2( laplacian_img.size(), CV_8UC4 );// = laplacian_img.clone();
	double min_val, max_val;
	cv::minMaxIdx( laplacian_img, &min_val, &max_val );
	laplacian_img.convertTo( prepared_img, CV_8UC1, 1.0 );
	cv::equalizeHist( prepared_img, prepared_img ); // Only works on 8-bit data
	//cv::erode( prepared_img, prepared_img, dilation_kernel );
	//cv::dilate( prepared_img, prepared_img, dilation_kernel );
	cv::medianBlur( prepared_img, prepared_img, 5 );
	//cv::dilate( prepared_img, prepared_img, dilation_kernel );
	double min_val2, max_val2;
	cv::minMaxIdx( prepared_img, &min_val2, &max_val2 );
	cv::cvtColor( prepared_img, prepared_img2, COLOR_GRAY2BGRA );
	//cv::normalize( prepared_img2, prepared_img2, 0, 255.0, NORM_MINMAX );

	//auto test = prepared_img.at<char>( 0, 0, 0 );
	//prepared_img.convertTo( prepared_img, CV_8UC3 );
	//imshow( "Test", prepared_img );
	//imshow( "test", prepared_img2 );
	emit Display_Image( prepared_img2 );
	return;
	//static int test_here = 0;
	//if( test_here++ < 10 )
		//return;
	this->current_image = camera->Get_Image();
	if( this->current_image.empty() )
		return;
	this->frame_passthrough->frame_to_use = this->current_image;

	Mat result;
	superRes->nextFrame( result );
	emit Display_Image( result );
	return;

	const Mat new_image = this->current_image;
	if( !new_image.data )
	{
		return;
	}

	//Mat new_image_grayscale;
	//cvtColor( new_image, new_image_grayscale, cv::COLOR_BGR2GRAY );
	//Mat output_test;
	//cvtColor( new_image_grayscale, output_test, cv::COLOR_GRAY2BGR );

	//emit Display_Image( output_test );
	//return;

	if( Overall_Image.cols == 0 )
	{
		Mat original_cuda_img( new_image );
		detector->detectAndCompute( original_cuda_img, Mat(), all_keypoints, all_descriptors );
		if( all_descriptors.cols < 3 )
			return;
		new_image.copyTo( Overall_Image );
		emit Display_Image( this->camera->Get_Image() );
		return;
	}

	Mat cuda_img1{ new_image };

	std::vector<KeyPoint> keypoints_1;
	Mat descriptors_1; // descriptors (features)
	detector->detectAndCompute( cuda_img1, Mat(), keypoints_1, descriptors_1 );

	std::vector< DMatch > matches;
	if( descriptors_1.cols == 0 )
		return;
	this->matcher->match( descriptors_1, all_descriptors, matches );

	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < matches.size(); i++ )
	{
		double dist = matches[ i ].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	//printf( "-- Max dist : %f \n", max_dist );
	//printf( "-- Min dist : %f \n", min_dist );
	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitrary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;
	for( int i = 0; i < matches.size(); i++ )
	{
		//if( matches[ i ].distance <= max( 1.2 * min_dist, 0.05 ) )
		if( matches[ i ].distance <= max( 2.0 * min_dist, 0.05 ) )
		{
			good_matches.push_back( matches[ i ] );
		}
	}
	std::sort( good_matches.begin(), good_matches.end(), []( DMatch x, DMatch  y ) { return x.distance < y.distance; } );

	if( good_matches.size() >= 3 )
	{
		Point2f reference_points[ 3 ];
		Point2f new_points[ 3 ];
		for( int i = 0; i < 3; i++ )
		{
			reference_points[ i ] = keypoints_1[ good_matches[ i ].queryIdx ].pt;
			new_points[ i ] = all_keypoints[ good_matches[ i ].trainIdx ].pt;
		}
		Mat warp_mat = getAffineTransform( reference_points, new_points );
		Mat warped_image;
		warpAffine( new_image, warped_image, warp_mat, new_image.size() );
		//emit Display_Image( warped_image );

		Mat img_matches;
		drawMatches( new_image, keypoints_1, Overall_Image, all_keypoints,
					 std::vector< DMatch >( &good_matches[0], &good_matches[ 0 ] + 3 ), img_matches, Scalar::all( -1 ), Scalar::all( -1 ),
					 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners( 4 );
		obj_corners[ 0 ] = cv::Point( 0, 0 ); obj_corners[ 1 ] = cv::Point( new_image.cols, 0 );
		obj_corners[ 2 ] = cv::Point( new_image.cols, new_image.rows ); obj_corners[ 3 ] = cv::Point( 0, new_image.rows );
		std::vector<Point2f> scene_corners( 4 );
		//perspectiveTransform( obj_corners, scene_corners, H );
		//warpAffine( obj_corners, scene_corners, warp_mat, cv::Size( 2, 4 ) );
		////-- Draw lines between the corners (the mapped object in the scene - image_2 )
		//line( img_matches, scene_corners[ 0 ] + Point2f( new_image.cols, 0 ), scene_corners[ 1 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
		//line( img_matches, scene_corners[ 1 ] + Point2f( new_image.cols, 0 ), scene_corners[ 2 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
		//line( img_matches, scene_corners[ 2 ] + Point2f( new_image.cols, 0 ), scene_corners[ 3 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
		//line( img_matches, scene_corners[ 3 ] + Point2f( new_image.cols, 0 ), scene_corners[ 0 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
		//img_matches = new_image_aligned;

		emit Display_Image( img_matches );

	}

	if( 0 ) // Homography stuff
	{
		//-- Draw only "good" matches
		Mat img_matches;
		drawMatches( new_image, keypoints_1, Overall_Image, all_keypoints,
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
				scene.push_back( all_keypoints[ good_matches[ i ].trainIdx ].pt );
				Point2f location_delta = keypoints_1[ good_matches[ i ].queryIdx ].pt - all_keypoints[ good_matches[ i ].trainIdx ].pt;
				double distance = location_delta.dot( location_delta );
			}
			Mat H = cv::findHomography( obj, scene, cv::RANSAC );
			if( H.data )
			{
				vector<Mat> rotation, translation, normals;
				int solutions = cv::decomposeHomographyMat( H, cv::Mat::eye( H.size(), H.type() ), rotation, translation, normals );
				//int solutions = decomposeHomographyMat( homography, cameraMatrix, Rs_decomp, ts_decomp, normals_decomp );
				std::cout << "Homography:\n" << H << "\n";
				bool found_a_likely_decomposition = false;
				for( int i = 0; i < solutions; i++ )
				{
					if( normals[ i ].at<double>( 2 ) > 0 )
						continue;
					if( translation[ i ].data == nullptr )
					{
						if( cv::countNonZero( translation[ i ] != translation[ i ] ) > 0 ) // Don't use anything with NaN values
							continue;
					}
					Mat result = rotation[ i ] * Mat( Vec3d{ 0, 0, -1 } );
					auto test = result.at<double>( 2 );
					auto test2 = std::min( abs( result.at<double>( 2 ) ), 1.0 );
					auto test3 = std::acos( std::min( abs( result.at<double>( 2 ) ), 1.0 ) );
					double rotation_degrees = 180. / 3.1415926535 * std::acos( std::min( abs( result.at<double>( 2 ) ), 1.0 ) );
					if( rotation_degrees > 5 )
						continue;
					std::cout << "rotation:\n" << rotation_degrees << "\n";
					std::cout << "translation:\n" << translation[ i ] << "\n";
					std::cout << "normals:\n" << normals[ i ] << "\n";

					found_a_likely_decomposition = true;
				}
				Mat test;
				test.type();
				if( !found_a_likely_decomposition )
					return;
				//-- Get the corners from the image_1 ( the object to be "detected" )
				std::vector<Point2f> obj_corners( 4 );
				obj_corners[ 0 ] = cv::Point( 0, 0 ); obj_corners[ 1 ] = cv::Point( new_image.cols, 0 );
				obj_corners[ 2 ] = cv::Point( new_image.cols, new_image.rows ); obj_corners[ 3 ] = cv::Point( 0, new_image.rows );
				std::vector<Point2f> scene_corners( 4 );
				perspectiveTransform( obj_corners, scene_corners, H );
				Mat new_image_aligned;
				Overall_Image.copyTo( new_image_aligned );
				warpPerspective( new_image, new_image_aligned, H, Overall_Image.size() * 2, INTER_LINEAR + WARP_INVERSE_MAP );
				//-- Draw lines between the corners (the mapped object in the scene - image_2 )
				line( img_matches, scene_corners[ 0 ] + Point2f( new_image.cols, 0 ), scene_corners[ 1 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
				line( img_matches, scene_corners[ 1 ] + Point2f( new_image.cols, 0 ), scene_corners[ 2 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
				line( img_matches, scene_corners[ 2 ] + Point2f( new_image.cols, 0 ), scene_corners[ 3 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
				line( img_matches, scene_corners[ 3 ] + Point2f( new_image.cols, 0 ), scene_corners[ 0 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
				//img_matches = new_image_aligned;

				emit Display_Image( img_matches );
			}

		}
	}
}

void Live_Stitcher::GPU_Stitch_Loop()
{
	//return;
	const Mat new_image = camera->Get_Image();
	//const Mat new_image = cv::Mat( 1920, 1080, CV_8UC3 );
	if( !new_image.data )
	{
		return;
	}

	//QThread::sleep( 1 );
	Mat new_image_grayscale;
	cvtColor( new_image, new_image_grayscale, cv::COLOR_BGR2GRAY );
	Mat output_test;
	cvtColor( new_image_grayscale, output_test, cv::COLOR_GRAY2BGR );

	//emit Display_Image( output_test );
	//return;

	if( Overall_Image.cols == 0 )
	{
		cuda::GpuMat original_cuda_img( new_image_grayscale );
		detector->detectAndCompute( original_cuda_img, cuda::GpuMat(), all_keypoints, gpu_all_descriptors );
		if( gpu_all_descriptors.cols < 3 )
			return;
		new_image.copyTo( Overall_Image );
		emit Display_Image( this->camera->Get_Image() );
		return;
	}

	cuda::GpuMat cuda_img1{ new_image_grayscale };

	std::vector<KeyPoint> keypoints_1;
	cuda::GpuMat descriptors_1; // descriptors (features)
	detector->detectAndCompute( cuda_img1, cuda::GpuMat(), keypoints_1, descriptors_1 );

	std::vector< DMatch > matches;
	if( descriptors_1.cols == 0 )
		return;
	gpu_matcher->match( descriptors_1, gpu_all_descriptors, matches );

	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < matches.size(); i++ )
	{
		double dist = matches[ i ].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	//printf( "-- Max dist : %f \n", max_dist );
	//printf( "-- Min dist : %f \n", min_dist );
	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitrary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;
	for( int i = 0; i < matches.size(); i++ )
	{
		//if( matches[ i ].distance <= max( 1.2 * min_dist, 0.05 ) )
		if( matches[ i ].distance <= max( 2.0 * min_dist, 0.05 ) )
		{
			good_matches.push_back( matches[ i ] );
		}
	}

	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches( new_image, keypoints_1, Overall_Image, all_keypoints,
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
			scene.push_back( all_keypoints[ good_matches[ i ].trainIdx ].pt );
			Point2f location_delta = keypoints_1[ good_matches[ i ].queryIdx ].pt - all_keypoints[ good_matches[ i ].trainIdx ].pt;
			double distance = location_delta.dot( location_delta );
		}
		Mat H = cv::findHomography( obj, scene, cv::RANSAC );
		if( H.data )
		{
			vector<Mat> rotation, translation, normals;
			int solutions = cv::decomposeHomographyMat( H, cv::Mat::eye( H.size(), H.type() ), rotation, translation, normals );
			//int solutions = decomposeHomographyMat( homography, cameraMatrix, Rs_decomp, ts_decomp, normals_decomp );
			std::cout << "Homography:\n" << H << "\n";
			bool found_a_likely_decomposition = false;
			for( int i = 0; i < solutions; i++ )
			{
				if( normals[ i ].at<double>( 2 ) > 0 )
					continue;
				if( translation[ i ].data == nullptr )
				{
					if( cv::countNonZero( translation[ i ] != translation[ i ] ) > 0 ) // Don't use anything with NaN values
						continue;
				}
				Mat result = rotation[ i ] * Mat( Vec3d{ 0, 0, -1 } );
				auto test = result.at<double>( 2 );
				auto test2 = std::min( abs( result.at<double>( 2 ) ), 1.0 );
				auto test3 = std::acos( std::min( abs( result.at<double>( 2 ) ), 1.0 ) );
				double rotation_degrees = 180. / 3.1415926535 * std::acos( std::min( abs( result.at<double>( 2 ) ), 1.0 ) );
				if( rotation_degrees > 5 )
					continue;
				std::cout << "rotation:\n" << rotation_degrees << "\n";
				std::cout << "translation:\n" << translation[ i ] << "\n";
				std::cout << "normals:\n" << normals[ i ] << "\n";

				found_a_likely_decomposition = true;
			}

			if( !found_a_likely_decomposition )
				return;

			//-- Get the corners from the image_1 ( the object to be "detected" )
			std::vector<Point2f> obj_corners( 4 );
			obj_corners[ 0 ] = cv::Point( 0, 0 ); obj_corners[ 1 ] = cv::Point( new_image.cols, 0 );
			obj_corners[ 2 ] = cv::Point( new_image.cols, new_image.rows ); obj_corners[ 3 ] = cv::Point( 0, new_image.rows );
			std::vector<Point2f> scene_corners( 4 );
			perspectiveTransform( obj_corners, scene_corners, H );
			Mat new_image_aligned;
			Overall_Image.copyTo( new_image_aligned );
			warpPerspective( new_image, new_image_aligned, H, Overall_Image.size() * 2, INTER_LINEAR + WARP_INVERSE_MAP );
			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line( img_matches, scene_corners[ 0 ] + Point2f( new_image.cols, 0 ), scene_corners[ 1 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
			line( img_matches, scene_corners[ 1 ] + Point2f( new_image.cols, 0 ), scene_corners[ 2 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
			line( img_matches, scene_corners[ 2 ] + Point2f( new_image.cols, 0 ), scene_corners[ 3 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
			line( img_matches, scene_corners[ 3 ] + Point2f( new_image.cols, 0 ), scene_corners[ 0 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
			//img_matches = new_image_aligned;

			emit Display_Image( img_matches );
		}

	}
}

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
		//int i = 1 and 2;
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
	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher( cv::NORM_HAMMING );

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
