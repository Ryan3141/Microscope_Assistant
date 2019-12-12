#pragma once

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
#include "opencv2/xfeatures2d/cuda.hpp"

namespace pcv // Pleasant_OpenCV
{
template<int Data_Type, int Num_Colors, bool On_Gpu = false>
class Image
{};

template<int Data_Type, int Num_Colors>
class Image<Data_Type, Num_Colors, false> : public cv::Mat
{};

template<int Data_Type, int Num_Colors>
class Image<Data_Type, Num_Colors, true> : public cv::cuda::GpuMat
{};

using Gray_Float_Image = Image<CV_64F, 1, false>;
using RGB_Float_Image = Image<CV_64F, 3, false>;
using BGR_Float_Image = Image<CV_64F, 3, false>;
using RGBA_Float_Image = Image<CV_64F, 4, false>;
using BGRA_Float_Image = Image<CV_64F, 4, false>;

using Gray_UChar_Image = Image<CV_8U, 1, false>;
using RGB_UChar_Image = Image<CV_8U, 3, false>;
using BGR_UChar_Image = Image<CV_8U, 3, false>;
using RGBA_UChar_Image = Image<CV_8U, 4, false>;
using BRGA_UChar_Image = Image<CV_8U, 4, false>;

using Gray_Float_Image_Gpu = Image<CV_64F, 1, true>;
using RGB_Float_Image_Gpu = Image<CV_64F, 3, true>;
using BGR_Float_Image_Gpu = Image<CV_64F, 3, true>;
using RGBA_Float_Image_Gpu = Image<CV_64F, 4, true>;
using BGRA_Float_Image_Gpu = Image<CV_64F, 4, true>;

using Gray_UChar_Image_Gpu = Image<CV_8U, 1, true>;
using RGB_UChar_Image_Gpu = Image<CV_8U, 3, true>;
using BGR_UChar_Image_Gpu = Image<CV_8U, 3, true>;
using RGBA_UChar_Image_Gpu = Image<CV_8U, 4, true>;
using BRGA_UChar_Image_Gpu = Image<CV_8U, 4, true>;

template<int Data_Type, int Num_Colors, bool On_Gpu, int Data_Type2, int Num_Colors2, bool On_Gpu2>
inline void Laplacian( const Image<Data_Type, Num_Colors, On_Gpu> & input,
					   Image<Data_Type2, Num_Colors2, On_Gpu2> & output,
					   int ksize = 1, double scale = 1, double delta = 0,
					   int borderType = BORDER_DEFAULT )
{
	static_assert(Data_Type2 == CV_64F, "Output datatype for " __FUNCTION__ " must be float datatype");
	static_assert(On_Gpu == On_Gpu2, "Datatypes in " __FUNCTION__ " cannot have one image on gpu and one not");
	static_assert(Num_Colors == Num_Colors2, "Datatypes in " __FUNCTION__ " must have the same number of colors (or both be grayscale)");
	//static_assert(Output_Type::data_type == CV_64F, "Output datatype must be float datatype");
	if( output.data == nullptr )
		output.create( input.size(), CV_MAKETYPE( Data_Type2, Num_Colors2 ) );
	if( On_Gpu )
	{
		cv::cuda::Stream stream;
		cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createLaplacianFilter(
			CV_MAKETYPE( Data_Type, Num_Colors ), CV_MAKETYPE( Data_Type2, Num_Colors2 ), ksize, scale );
		filter->apply( static_cast<cv::cuda::GpuMat>(input), static_cast<cv::cuda::GpuMat>(output), stream );
	}
	else
	{
		cv::Laplacian( static_cast<cv::Mat>(input), static_cast<cv::Mat>(output), CV_MAKETYPE( Data_Type2, Num_Colors2 ),
					   ksize, scale, delta, borderType );
	}
}
template<int Data_Type, int Num_Colors, bool On_Gpu, int Data_Type2, int Num_Colors2, bool On_Gpu2>
inline void Convert( const Image<Data_Type, Num_Colors, On_Gpu> & input, Image<Data_Type2, Num_Colors2, On_Gpu2> & output )
{
	static_assert( false, __FUNCTION__ " not yet implemented");
	static_assert(On_Gpu == On_Gpu2, "Datatypes in " __FUNCTION__ " cannot have one image on gpu and one not");
	static_assert(!(On_Gpu || On_Gpu2), __FUNCTION__ " not yet implemented on gpu");
	if( output.data == nullptr )
		output.create( input.size(), CV_MAKETYPE( Data_Type2, Num_Colors2 ) );
	if( On_Gpu )
	{

		static_cast<cv::cuda::GpuMat>(input).convertTo( static_cast<cv::cuda::GpuMat>(input),
														CV_MAKETYPE( Data_Type2, Num_Colors2 ) );
	}
	else
	{
		static_cast<cv::Mat>(input).convertTo( static_cast<cv::Mat>(input),
											   CV_MAKETYPE( Data_Type2, Num_Colors2 ) );
	}
}

template<int Data_Type, int Num_Colors, bool On_Gpu, int Data_Type2, int Num_Colors2, bool On_Gpu2>
inline void Change_Data_Type( const Image<Data_Type, Num_Colors, On_Gpu> & input, Image<Data_Type2, Num_Colors2, On_Gpu2> & output )
{
	static_assert(On_Gpu == On_Gpu2, "Datatypes in " __FUNCTION__ " cannot have one image on gpu and one not");
	static_assert(Num_Colors == Num_Colors2, "Input and output in " __FUNCTION__ " must have the same number of color channels");
	if( output.data == nullptr )
		output.create( input.size(), CV_MAKETYPE( Data_Type2, Num_Colors2 ) );
	if( On_Gpu )
	{

		static_cast<cv::cuda::GpuMat>(input).convertTo( static_cast<cv::cuda::GpuMat>(output),
														CV_MAKETYPE( Data_Type2, Num_Colors2 ) );
	}
	else
	{
		static_cast<cv::Mat>(input).convertTo( static_cast<cv::Mat>(output),
											   CV_MAKETYPE( Data_Type2, Num_Colors2 ) );
	}
}

template<int c, int Data_Type, int Num_Colors, bool On_Gpu, int Data_Type2, int Num_Colors2, bool On_Gpu2>
inline void cvtColor( const Image<Data_Type, Num_Colors, On_Gpu> & input, Image<Data_Type2, Num_Colors2, On_Gpu2> & output )
{
	static_assert(Data_Type == Data_Type2, "Datatypes in " __FUNCTION__ " must be the same datatype (uchar, float, etc)");
	static_assert(On_Gpu == On_Gpu2, "Datatypes in " __FUNCTION__ " cannot have one image on gpu and one not");
	static_assert(!(On_Gpu || On_Gpu2), __FUNCTION__ " not yet implemented on gpu");
	constexpr bool input_requested_1_channel = c == COLOR_GRAY2RGB || c == COLOR_GRAY2BGR || c == COLOR_GRAY2RGBA || c == COLOR_GRAY2BGRA;
	constexpr bool input_requested_3_channels = c == COLOR_RGB2GRAY || c == COLOR_BGR2GRAY || c == COLOR_RGB2BGR || c == COLOR_BGR2RGB || c == COLOR_RGB2RGBA || c == COLOR_BGR2RGBA || c == COLOR_RGB2BGRA || c == COLOR_BGR2BGRA;
	constexpr bool input_requested_4_channels = c == COLOR_RGBA2GRAY || c == COLOR_BGRA2GRAY || c == COLOR_RGBA2BGR || c == COLOR_BGRA2BGR || c == COLOR_RGBA2RGB || c == COLOR_BGRA2RGB || c == COLOR_RGBA2BGRA || c == COLOR_BGRA2RGBA;
	static_assert(input_requested_1_channel && Num_Colors == 1 || input_requested_3_channels && Num_Colors == 3 || input_requested_4_channels && Num_Colors == 4,
				   "Input number of color channels in " __FUNCTION__ " doesn't match requested conversion");
	constexpr bool output_requested_1_channel = c == COLOR_BGR2GRAY || c == COLOR_RGB2GRAY || c == COLOR_BGRA2GRAY || c == COLOR_RGBA2GRAY;
	constexpr bool output_requested_3_channels = c == COLOR_GRAY2BGR || c == COLOR_GRAY2RGB || c == COLOR_RGBA2BGR || c == COLOR_RGBA2RGB || c == COLOR_BGRA2BGR || c == COLOR_BGRA2RGB || c == COLOR_RGB2BGR || c == COLOR_BGR2RGB;
	constexpr bool output_requested_4_channels = c == COLOR_GRAY2BGRA || c == COLOR_GRAY2RGBA || c == COLOR_RGB2BGRA || c == COLOR_RGB2RGBA || c == COLOR_BGR2BGRA || c == COLOR_BGR2RGBA || c == COLOR_BGRA2RGBA || c == COLOR_RGBA2BGRA;
	static_assert(output_requested_1_channel && Num_Colors2 == 1 || output_requested_3_channels && Num_Colors2 == 3 || output_requested_4_channels && Num_Colors2 == 4,
				   "Output number of color channels in " __FUNCTION__ " doesn't match requested conversion");
	if( output.data == nullptr )
		output.create( input.size(), CV_MAKETYPE( Data_Type2, Num_Colors2 ) );
	if( On_Gpu )
	{
	}
	else
	{
		cv::cvtColor( static_cast<cv::Mat>(input), static_cast<cv::Mat>(output), c, Num_Colors2 );
	}
}

template<int Data_Type, int Num_Colors, bool On_Gpu,
	int Data_Type2, int Num_Colors2, bool On_Gpu2,
	int Data_Type3, int Num_Colors3, bool On_Gpu3>
inline void multiply( const Image<Data_Type, Num_Colors, On_Gpu> & src1,
					  const Image<Data_Type2, Num_Colors2, On_Gpu2> & src2,
					  Image<Data_Type3, Num_Colors3, On_Gpu3> & src3,
					  double scale = 1 )
{
	static_assert(Data_Type == Data_Type2 && Data_Type == Data_Type3, "Datatypes in " __FUNCTION__ " must be the same datatype (uchar, float, etc)");
	static_assert(On_Gpu == On_Gpu2 && On_Gpu == On_Gpu3, "Datatypes in " __FUNCTION__ " cannot have one image on gpu and one not");
	static_assert(Num_Colors == Num_Colors2 && Num_Colors == Num_Colors3, "Datatypes in " __FUNCTION__ " must have the same number of colors (or both be grayscale)");
	//static_assert(Output_Type::data_type == CV_64F, "Output datatype must be float datatype");
	if( output.data == nullptr )
		output.create( src3.size(), CV_MAKETYPE( Data_Type3, Num_Colors3 ) );
	if( On_Gpu )
	{
		cv::cuda::multiply( static_cast<cv::cuda::GpuMat>(src1), static_cast<cv::cuda::GpuMat>(src2),
					  static_cast<cv::cuda::GpuMat>(src3), scale );
	}
	else
	{
		cv::multiply( static_cast<cv::Mat>(src1), static_cast<cv::Mat>(src2),
					  static_cast<cv::Mat>(src3), scale );
	}
}

template<typename Input_Type, typename Output_Type>
inline void equalizeHist( const Input_Type & input, Output_Type & output )
{
	static_assert(On_Gpu == On_Gpu2, "Datatypes in " __FUNCTION__ " cannot have one image on gpu and one not");
	static_assert(!(On_Gpu || On_Gpu2), __FUNCTION__ " not yet implemented on gpu");
	if( output.data == nullptr )
		output.create( input.size(), CV_MAKETYPE( Data_Type2, Num_Colors2 ) );
	if( On_Gpu )
	{
	}
	else
	{
	}
}

};

