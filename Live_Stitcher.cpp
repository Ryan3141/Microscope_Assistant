#include "Live_Stitcher.h"

#include <QTimer>
#include <QFileInfo>
/*
* @file SURF_FlannMatcher
* @brief SURF detector + descriptor + FLANN Matcher
* @author A. Huaman
*/
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <fstream>
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

double Find_Blurriness( pcv::RGBA_UChar_Image & input );

class Image_Living_Properties
{
	bool Is_Image_Sharp( pcv::RGBA_UChar_Image & input );
};

static void Construct_Alpha_Blend_Image( cv::Mat & alpha_mask, cv::Size image_size )
{
	std::ifstream alpha_file( "Test.csv" );
	//alpha_mask.create( Size( 257, 193 ), CV_8UC1 );
	alpha_mask.create( cv::Size( 257, 193 ), CV_32FC1 );
	for( int y = -1; y < alpha_mask.rows + 1; y++ )
	{
		for( int x = -1; x < alpha_mask.cols + 1; x++ )
		{
			if( x < 0 || y < 0 || x >= alpha_mask.cols || y >= alpha_mask.rows )
				continue;
			//unsigned char* output_combined_pointer = &(alpha_mask.at<float>( y, x ));
			float* output_combined_pointer = &(alpha_mask.at<float>( y, x ));
			float zero_to_one;
			alpha_file >> zero_to_one;
			//*output_combined_pointer = (unsigned char)min( max( 0, (int)(zero_to_one * 255) ), 255 );
			*output_combined_pointer = std::min( std::max( 0.f, (zero_to_one) ), 255.f );
		}
	}
	alpha_file.close();
	if( image_size == Size( 0, 0 ) )
		alpha_mask.create( cv::Size( 0, 0 ), CV_32FC1 );
	else
		cv::resize( alpha_mask, alpha_mask, image_size );// alpha_mask.size() * 10 );
}

void Add_With_Alpha_Ratio( pcv::RGBA_UChar_Image & output_combined, const pcv::RGBA_UChar_Image & to_be_added, cv::Rect roi, bool is_alpha_present, cv::Mat & alpha_mask )
{
	if( alpha_mask.size() != to_be_added.size() )
		Construct_Alpha_Blend_Image( alpha_mask, to_be_added.size() );// Size( 2576, 1932 ) );

	for( int y = 0; y < roi.height; y++ )
	{
		for( int x = 0; x < roi.width; x++ )
		{
			unsigned char* output_combined_pointer = &(output_combined.at<cv::Vec4b>( roi.y + y, roi.x + x )[ 0 ]);
			const unsigned char* to_be_added_pointer;
			double alpha;

			if( is_alpha_present )
			{
				to_be_added_pointer = &(to_be_added.at<cv::Vec4b>( y, x )[ 0 ]);
				alpha = *(to_be_added_pointer + 3);
			}
			else
			{
				to_be_added_pointer = &(to_be_added.at<cv::Vec4b>( y, x )[ 0 ]);
				//double x_distance = double( x - roi.width / 2 ) / (roi.width / 2);
				//double y_distance = double( y - roi.height / 2 ) / (roi.height / 2);
				//double distance_from_center = x_distance * x_distance + y_distance * y_distance;
				////double falloff_function = 1 / (1 + exp( -10.0 * (distance_from_center - 1.0) ));
				//double falloff_function = 1 - distance_from_center;
				//alpha = max( 1., 255 * max( 0.0, min( 1.0, falloff_function ) ) );
				alpha = 255 * alpha_mask.at<float>( y, x );
			}

			double original_alpha = (double)*(output_combined_pointer + 3);
			if( original_alpha + alpha == 0 )
				continue; // If the alphas are zero, there is nothing to add
			double amount_of_original = original_alpha / (original_alpha + alpha);
			double amount_of_new = 1 - amount_of_original;

			*(output_combined_pointer) = amount_of_original * *(output_combined_pointer)+amount_of_new * *(to_be_added_pointer);
			*(output_combined_pointer + 1) = amount_of_original * *(output_combined_pointer + 1) + amount_of_new * *(to_be_added_pointer + 1);
			*(output_combined_pointer + 2) = amount_of_original * *(output_combined_pointer + 2) + amount_of_new * *(to_be_added_pointer + 2);
			*(output_combined_pointer + 3) = std::max( original_alpha, alpha );
		}
	}
}

pcv::RGBA_UChar_Image Add_To_Picture( const pcv::RGBA_UChar_Image overall_image, const pcv::RGBA_UChar_Image new_image, const cv::Point & offset, cv::Point & shifting_origin, cv::Mat & alpha_mask )
{
	auto background_color = cv::Scalar( 255, 255, 255, 0 );
	cv::Point relative_to_origin = offset + shifting_origin;

	// Recreate the overall image so that we can paste the previous version shifted due to the potential new origin location
	pcv::RGBA_UChar_Image combined(
					Mat( std::max( overall_image.rows, new_image.rows + relative_to_origin.y ) - std::min( 0, relative_to_origin.y ),
					  std::max( overall_image.cols, new_image.cols + relative_to_origin.x ) - std::min( 0, relative_to_origin.x ), overall_image.type(), background_color ) );
	cv::Point new_shift_in_origin = cv::Point( std::min( 0, relative_to_origin.x ), std::min( 0, relative_to_origin.y ) );
	cv::Point location_of_new_image = relative_to_origin - new_shift_in_origin;
	cv::Rect roi1( -new_shift_in_origin, overall_image.size() );
	cv::Rect roi2( location_of_new_image, new_image.size() );

	Add_With_Alpha_Ratio( combined, overall_image, roi1, true, alpha_mask );
	Add_With_Alpha_Ratio( combined, new_image, roi2, false, alpha_mask );

	//overall_image.copyTo( combined( roi1 ) );
	//new_image.copyTo( combined( roi2 ) );

	shifting_origin -= new_shift_in_origin;
	return combined;
}

static void Keep_Inside( Rect & area, const Mat & image_to_keep_inside )
{
	int cols = image_to_keep_inside.cols;
	int rows = image_to_keep_inside.rows;
	int original_x = area.x;
	int original_y = area.y;
	area.x = min( cols - 1, max( 0, original_x ) );
	area.width = min( cols - area.x, min( area.width + original_x, area.width ) );
	area.y = min( rows - 1, max( 0, original_y ) );
	area.height = min( rows - area.y, min( area.height + original_y, area.height ) );
}

Point Find( const Mat & within_this, const Mat & find_this )
{
	Mat temp;
	matchTemplate( within_this, find_this, temp, TM_CCOEFF_NORMED );
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

	return maxLoc;
}

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
		this->gpu_detector = cuda::ORB::create();
		//this->gpu_detector = cuda::ORB::create( 500, 1.2f, 8, 31, 0, 2, 0, 31, 20, true );

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

	QTimer::singleShot( 0, this, &Live_Stitcher::Stitch_Loop ); // Start stitching loop
}

void Live_Stitcher::Find_Details_Mask( pcv::RGBA_UChar_Image & input, pcv::Gray_Float_Image & output ) const
{
	pcv::Gray_Float_Image grayscale;
	pcv::Convert<COLOR_RGBA2GRAY>( input, grayscale );

	pcv::Gray_Float_Image laplacian_img;
	pcv::Laplacian( grayscale, laplacian_img, 7, 1.0, 0.0, BORDER_REFLECT_101 );

	pcv::multiply( laplacian_img, laplacian_img, laplacian_img, 1.0 );

	{
		int threshold_value = 100000;
		int max_BINARY_value = 0; // Not used
		cv::threshold( laplacian_img, laplacian_img, threshold_value, max_BINARY_value, cv::THRESH_TRUNC );
		double min_val, max_val;
		cv::minMaxIdx( laplacian_img, &min_val, &max_val );
		pcv::normalize( laplacian_img, laplacian_img, 0, 255.0, NORM_MINMAX );
	}
	{
		constexpr int erosion_size = 2;
		Mat dilation_kernel = getStructuringElement( cv::MORPH_ELLIPSE,
													 Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ),
													 Point( erosion_size, erosion_size ) );
		cv::erode( laplacian_img, laplacian_img, dilation_kernel );
		cv::medianBlur( laplacian_img, laplacian_img, 3 );
		//cv::dilate( laplacian_img, laplacian_img, dilation_kernel );
	}
	if( false )
	{
		constexpr int dilate_size = 2;
		Mat dilation_kernel = getStructuringElement( cv::MORPH_ELLIPSE,
													 Size( 2 * dilate_size + 1, 2 * dilate_size + 1 ),
													 Point( dilate_size, dilate_size ) );
		cv::dilate( laplacian_img, laplacian_img, dilation_kernel );
	}

	constexpr int blur_size = 11;
	cv::GaussianBlur( laplacian_img, laplacian_img, Size( blur_size, blur_size ), 5.0, 5.0 );
	output = laplacian_img;
}

#if 0
{
	// process 1st image
	GpuMat imgGray1;  // load this with your grayscale image
	GpuMat keys1; // this holds the keys detected
	GpuMat desc1; // this holds the descriptors for the detected keypoints
	GpuMat mask1; // this holds any mask you may want to use, or can be replace by noArray() in the call below if no mask is needed
	vector<KeyPoint> cpuKeys1;  // holds keypoints downloaded from gpu

	//ADD CODE TO LOAD imgGray1

	orb->detectAndComputeAsync( imgGray1, mask1, keys1, desc1, false, m_stream );
	stream.waitForCompletion();
	orb->convert( keys1, cpuKeys1 ); // download keys to cpu if needed for anything...like displaying or whatever

	// process 2nd image
	GpuMat imgGray2;  // load this with your grayscale image
	GpuMat keys2; // this holds the keys detected
	GpuMat desc2; // this holds the descriptors for the detected keypoints
	GpuMat mask2; // this holds any mask you may want to use, or can be replace by noArray() in the call below if no mask is needed
	vector<KeyPoint> cpuKeys2;  // holds keypoints downloaded from gpu

	//ADD CODE TO LOAD imgGray2

	orb->detectAndComputeAsync( imgGray2, mask2, keys2, desc2, false, m_stream );
	stream.waitForCompletion();
	orb->convert( keys2, cpuKeys2 ); // download keys to cpu if needed for anything...like displaying or whatever

	if( desc2.rows > 0 )
	{
		vector<vector<DMatch>> cpuKnnMatches;
		GpuMat gpuKnnMatches;  // holds matches on gpu
		matcher->knnMatchAsync( desc2, desc1, gpuKnnMatches, 2, noArray(), *stream );  // find matches
		stream.waitForCompletion();

		matcher->knnMatchConvert( gpuKnnMatches, cpuKnnMatches ); // download matches from gpu and put into vector<vector<DMatch>> form on cpu

		vector<DMatch> matches;         // vector of good matches between tested images

		for( std::vector<std::vector<cv::DMatch> >::const_iterator it = cpuKnnMatches.begin(); it != cpuKnnMatches.end(); ++it )
		{
			if( it->size() > 1 && (*it)[ 0 ].distance / (*it)[ 1 ].distance < m_nnr )
			{  // use Nearest-Neighbor Ratio to determine "good" matches
				DMatch m = (*it)[ 0 ];
				matches.push_back( m );       // save good matches here                           

			}
		}
	}
}
#endif

template<class Image_Type, class DetectorType>
bool Get_Descriptors( Image_Match_Info<Image_Type> & image_to_look_at, cv::Ptr<DetectorType> detector )
{
	if( image_to_look_at.img.cols == 0 )
		return false;
	cv::resize( image_to_look_at.img, image_to_look_at.scaled_down_img, image_to_look_at.img.size() / Image_Match_Info<Image_Type>::scale );
	//cuda::GpuMat keys2; // this holds the keys detected
	//cuda::GpuMat desc2; // this holds the descriptors for the detected keypoints
	//vector<KeyPoint> cpuKeys2;  // holds keypoints downloaded from gpu
	////detector->detectAndComputeAsync( image_to_look_at.scaled_down_img, noArray(), keys2, desc2, false, m_stream );
	//detector->detectAndCompute( image_to_look_at.scaled_down_img, noArray(), image_to_look_at.keypoints, desc2 );

	//orb->detectAndComputeAsync( imgGray2, mask2, keys2, desc2, false, m_stream );
	//stream.waitForCompletion();
	//orb->convert( keys2, image_to_look_at.keypoints ); // download keys to cpu if needed for anything...like displaying or whatever

	detector->detectAndCompute( image_to_look_at.scaled_down_img, Mat(), image_to_look_at.keypoints, image_to_look_at.descriptors );
	//gpu_detector->detectAndCompute( image_to_look_at.scaled_down_img, Mat(), image_to_look_at.keypoints, image_to_look_at.descriptors );
	if( image_to_look_at.descriptors.cols < 3 )
		return false;
	else
		return true;
}

template<class Image_Type>
bool Find_Alignment( const Image_Match_Info<Image_Type> & main_image, const Image_Match_Info<Image_Type> & aligning_image,
					 cv::Ptr<cv::BFMatcher> matcher, Point2f & point_in_main, Point2f & point_in_aligning, std::vector< DMatch > & good_matches )
{
	std::vector< DMatch > matches;
	matcher->match( aligning_image.descriptors, main_image.descriptors, matches );

	double max_dist = 0;
	double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < matches.size(); i++ )
	{
		double dist = matches[ i ].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	//std::vector< DMatch > good_matches;
	double good_max_dist = 0;
	double good_min_dist = 100;
	for( int i = 0; i < matches.size(); i++ )
	{
		//if( matches[ i ].distance <= max( 1.2 * min_dist, 0.05 ) )
		if( matches[ i ].distance <= max( 2.0 * min_dist, 0.05 ) )
		{
			good_matches.push_back( matches[ i ] );
			double dist = matches[ i ].distance;
			if( dist < good_min_dist ) good_min_dist = dist;
			if( dist > good_max_dist ) good_max_dist = dist;
		}
		}
	
	if( good_matches.size() < 3 )
		return false;

	std::sort( good_matches.begin(), good_matches.end(), []( DMatch x, DMatch  y ) { return x.distance < y.distance; } );
	point_in_main = main_image.keypoints[ good_matches[ 0 ].trainIdx ].pt * Image_Match_Info<Image_Type>::scale;
	point_in_aligning = aligning_image.keypoints[ good_matches[ 0 ].queryIdx ].pt * Image_Match_Info<Image_Type>::scale;

	if( false )
	{ // Find average displacement
		Point2f weighted_center( 0.0, 0.0 );
		double total_weight = 0;
		for( int i = 0; i < good_matches.size(); i++ )
		{
			Point2f point_in_aligning = aligning_image.keypoints[ good_matches[ i ].queryIdx ].pt;
			Point2f point_in_main = main_image.keypoints[ good_matches[ i ].trainIdx ].pt;
			Point2f displacement = point_in_aligning - point_in_main;
			double weight = 1;
			if( good_max_dist - good_min_dist )
				weight = (good_max_dist - good_matches[ i ].distance) / (good_max_dist - good_min_dist); // Smaller distance gives larger weight
			total_weight += weight;
			weighted_center += weight * displacement;
		}

		Point2f average_displacement = -Image_Match_Info<Image_Type>::scale * weighted_center / total_weight;
	}

	if( false ) // Find affine transfer
	{
		std::array<Point2f, 3> reference_points;
		std::array<Point2f, 3> new_points;
		for( int i = 0; i < 3; i++ )
		{
			reference_points[ i ] = aligning_image.keypoints[ good_matches[ i ].queryIdx ].pt;
			new_points[ i ] = main_image.keypoints[ good_matches[ i ].trainIdx ].pt;
		}

		Mat alignment_matrix = getAffineTransform( reference_points, new_points );
	}
	return true;
}

void Live_Stitcher::Save_Image_And_Start_Over( QFileInfo path_to_file )
{
	imwrite( path_to_file.absoluteFilePath().toStdString().c_str(), Overall_Image.img );
	std::cout << path_to_file.absoluteFilePath().toStdString() << " saved\n";
	Reset_Stitching();
}

void Live_Stitcher::Reset_Stitching()
{
	Overall_Image = {};
	previous_image = {};
	previous_image_offset_in_overall_image = {};
	proposed_image = {};
	proposed_offset_in_overall_image = {};

	emit Display_Image( pcv::BGRA_UChar_Image{} );
}

void Live_Stitcher::Stitch_Image_And_Start_New()
{
	if( proposed_image.img.data == nullptr )
		return;

	Point shifted_origin( 0, 0 );
	Overall_Image.img = Add_To_Picture( Overall_Image.img, proposed_image.img, proposed_offset_in_overall_image, shifted_origin, alpha_mask );
	previous_image_offset_in_overall_image = proposed_offset_in_overall_image + shifted_origin;

	previous_image = proposed_image;
	previous_image.img = previous_image.img.clone();
	std::cout << "Stitched\n";
}

void Live_Stitcher::Stitch_Loop()
{
	this->current_image = camera->Get_Image();
	if( this->current_image.empty() )
	{
		QTimer::singleShot( 0, this, &Live_Stitcher::Stitch_Loop ); // Rerun this function on completion
		return;
	}

	if( false )
	{
		double blurriness = Find_Blurriness( this->current_image );
		std::cout << blurriness << std::endl;
		pcv::Gray_Float_Image mask;
		Find_Details_Mask( this->current_image, mask );

		pcv::RGBA_UChar_Image to_be_displayed;
		pcv::Convert<COLOR_GRAY2RGBA>( mask, to_be_displayed );
		//emit Display_Image( to_be_displayed );

		//pcv::equalizeHist( laplacian_img, laplacian_img ); // Only works on 8-bit data
	}
	//return;

	if( false )
	{
		this->current_image = camera->Get_Image();
		if( this->current_image.empty() )
			return;
		this->frame_passthrough->frame_to_use = this->current_image;

		Mat result;
		superRes->nextFrame( result );
		emit Display_Image( result );
		return;
	}

	//Mat new_image_grayscale;
	//cvtColor( new_image, new_image_grayscale, cv::COLOR_BGR2GRAY );
	//Mat output_test;
	//cvtColor( new_image_grayscale, output_test, cv::COLOR_GRAY2BGR );

	//emit Display_Image( output_test );
	//return;

	proposed_image = { this->current_image };
	{ // Skip for just overlays
		proposed_offset_in_overall_image = Point( 0, 0 );
		Point ignore_origin( 0, 0 );
		pcv::RGBA_UChar_Image combined_images = Add_To_Picture( Overall_Image.img, proposed_image.img, proposed_offset_in_overall_image, ignore_origin, alpha_mask );
		emit Display_Image( combined_images );
		QTimer::singleShot( 0, this, &Live_Stitcher::Stitch_Loop ); // Rerun this function on completion
		return;
	}
	if( !Get_Descriptors( previous_image, detector ) )
	{
		//previous_image.img = this->current_image.clone();
		//cv::Point ignore(0,0);
		//Overall_Image.img = Add_To_Picture( pcv::RGBA_UChar_Image(), previous_image.img, Point(0,0), ignore, alpha_mask );
		QTimer::singleShot( 0, this, &Live_Stitcher::Stitch_Loop ); // Rerun this function on completion
		return;
	}

	if( !Get_Descriptors( proposed_image, detector ) )
	{
		QTimer::singleShot( 0, this, &Live_Stitcher::Stitch_Loop ); // Rerun this function on completion
		return;
	}

	//Mat alignment_matrix;
	Point2f point_in_main;
	Point2f point_in_aligning;
	std::vector< DMatch > good_matches;
	if( !Find_Alignment( previous_image, proposed_image, this->matcher, point_in_main, point_in_aligning, good_matches ) )
	{
		QTimer::singleShot( 0, this, &Live_Stitcher::Stitch_Loop ); // Rerun this function on completion
		return;
	}

	{
		Rect search_area( Point2i( point_in_main ) - Point( previous_image.img.size() / 32 ), Size( previous_image.img.size() / 16 ) );
		Keep_Inside( search_area, previous_image.img );
		Rect search_pattern( Point( point_in_aligning ) - Point( proposed_image.img.size() / 64 ), Size( proposed_image.img.size() / 32 ) );
		Keep_Inside( search_pattern, proposed_image.img );
		Point additional_offset = Find( previous_image.img( search_area ), proposed_image.img( search_pattern ) );
		Point offset_between_images = search_area.tl() - search_pattern.tl() + additional_offset;
		if( false )// offset_between_images == Point( 0, 0 ) )
		{
			for( int x = 0; x < previous_image.img.cols; x++ )
			{
				for( int y = 0; y < previous_image.img.cols; y++ )
				{
					cv::Vec4b & previous_image_pixel = previous_image.img.at<cv::Vec4b>( y, x );
					cv::Vec4b & new_image_pixel = proposed_image.img.at<cv::Vec4b>( y, x );
					for( int i = 0; i < 4; i++ )
					{
						int weighted_sum = 0;
					}
				}
			}
		}
		else
			cleaner_image_counter = 1;
		//Mat warped_image;
		//warpAffine( this->current_image, warped_image, alignment_matrix, this->current_image.size() );
		proposed_offset_in_overall_image = offset_between_images + previous_image_offset_in_overall_image;
		Point ignore_origin( 0, 0 );
		pcv::RGBA_UChar_Image combined_images = Add_To_Picture( Overall_Image.img, proposed_image.img, proposed_offset_in_overall_image, ignore_origin, alpha_mask );
		//line( Overall_Image.img, search_area.tl() + shifting_origin + Point( -search_area.width / 2, search_area.height / 2 ), search_area.tl() + shifting_origin + Point( search_area.width / 2, search_area.height / 2 ), Scalar( 0, 255, 0 ), 4 );
		//line( Overall_Image.img, search_area.tl() + shifting_origin + Point( search_area.width / 2, -search_area.height / 2 ), search_area.tl() + shifting_origin + Point( search_area.width / 2, search_area.height / 2 ), Scalar( 0, 255, 0 ), 4 );
		//line( Overall_Image.img, search_area.tl() + shifting_origin + Point( -search_area.width / 2, -search_area.height / 2 ), search_area.tl() + shifting_origin + Point( search_area.width / 2, -search_area.height / 2 ), Scalar( 0, 255, 0 ), 4 );
		//line( Overall_Image.img, search_area.tl() + shifting_origin + Point( -search_area.width / 2, -search_area.height / 2 ), search_area.tl() + shifting_origin + Point( -search_area.width / 2, search_area.height / 2 ), Scalar( 0, 255, 0 ), 4 );
		//Overall_Image.img = Add_To_Picture( Previous_Image.img, aligning_image.img, Point( average_displacement ), shifting_origin, alpha_mask );
		emit Display_Image( combined_images );

		Mat img_matches;
		drawMatches( proposed_image.scaled_down_img, proposed_image.keypoints, previous_image.scaled_down_img, previous_image.keypoints,
					 std::vector< DMatch >( &good_matches[0], &good_matches[ 0 ] + std::min( 30, int(good_matches.size()) ) ),
					 img_matches, Scalar::all( -1 ), Scalar::all( -1 ),
					 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		//-- Get the corners from the image_1 ( the object to be "detected" )
		//std::vector<Point2f> obj_corners( 4 );
		//obj_corners[ 0 ] = cv::Point( 0, 0 );
		//obj_corners[ 1 ] = cv::Point( cuda_img1.cols, 0 );
		//obj_corners[ 2 ] = cv::Point( cuda_img1.cols, cuda_img1.rows );
		//obj_corners[ 3 ] = cv::Point( 0, cuda_img1.rows );
		//std::vector<Point2f> scene_corners( 4 );
		//perspectiveTransform( obj_corners, scene_corners, H );
		//warpAffine( obj_corners, scene_corners, warp_mat, cv::Size( 2, 4 ) );
		////-- Draw lines between the corners (the mapped object in the scene - image_2 )
		//line( img_matches, scene_corners[ 0 ] + Point2f( new_image.cols, 0 ), scene_corners[ 1 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
		//line( img_matches, scene_corners[ 1 ] + Point2f( new_image.cols, 0 ), scene_corners[ 2 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
		//line( img_matches, scene_corners[ 2 ] + Point2f( new_image.cols, 0 ), scene_corners[ 3 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
		//line( img_matches, scene_corners[ 3 ] + Point2f( new_image.cols, 0 ), scene_corners[ 0 ] + Point2f( new_image.cols, 0 ), Scalar( 0, 255, 0 ), 4 );
		//img_matches = new_image_aligned;

		emit Display_Debug_Image( img_matches );

	}

#if 0
	if( 0 ) // Homography stuff
	{
		//-- Draw only "good" matches
		Mat new_image = this->current_image;
		Mat img_matches;
		drawMatches( this->current_image, keypoints_1, Overall_Image, all_keypoints,
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
#endif

	QTimer::singleShot( 0, this, &Live_Stitcher::Stitch_Loop ); // Rerun this function on completion
}

void Live_Stitcher::GPU_Stitch_Loop()
{
#if 0
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
#endif
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

double Find_Blurriness( pcv::RGBA_UChar_Image & input )
{
	pcv::RGBA_Float_Image laplacian_img;
	pcv::Laplacian( input, laplacian_img, 7, 1.0, 0.0, BORDER_REFLECT_101 );
	pcv::multiply( laplacian_img, laplacian_img, laplacian_img, 1.0 );

	double blurriness = 0;
	cv::Scalar test = cv::mean( laplacian_img );
	for( int i = 0; i < 3; i++ )
		blurriness = test( i );

	return blurriness;
}

