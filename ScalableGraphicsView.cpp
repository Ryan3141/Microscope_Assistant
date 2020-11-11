#include "scalablegraphicsview.h"

#include <QMouseEvent>
#include <QScrollBar>
#include <QGraphicsItem>

#include <QImage>
#include <QPixmap>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

using namespace std;
using namespace cv;

//inline QImage  cvMatToQImage( const cv::Mat &inMat )
//{
//	switch( inMat.type() )
//	{
//		// 8-bit, 4 channel
//		case CV_8UC4:
//		{
//			QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB32 );
//
//			return image;
//		}
//
//		// 8-bit, 3 channel
//		case CV_8UC3:
//		{
//			QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB888 );
//
//			return image.rgbSwapped();
//		}
//
//		// 8-bit, 1 channel
//		case CV_8UC1:
//		{
//			static QVector<QRgb>  sColorTable;
//
//			// only create our color table once
//			if( sColorTable.isEmpty() )
//			{
//				for( int i = 0; i < 256; ++i )
//					sColorTable.push_back( qRgb( i, i, i ) );
//			}
//
//			QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_Indexed8 );
//
//			image.setColorTable( sColorTable );
//
//			return image;
//		}
//
//		default:
//		//qWarning() << "ASM::cvMatToQImage() - cv::Mat image type not handled in switch:" << inMat.type();
//		break;
//	}
//
//	return QImage();
//}
//
//inline QPixmap cvMatToQPixmap( const cv::Mat &inMat )
//{
//	return QPixmap::fromImage( cvMatToQImage( inMat ) );
//}

//Thanks to http://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview
ScalableGraphicsView::ScalableGraphicsView( QWidget *parent )
	: QGraphicsView( parent )
{
	this->_zoom = 0;
	this->_scene = new QGraphicsScene( this );
	//this->_photo = nullptr;// new QGraphicsPixmap();
	this->_photo_handle = nullptr;
	//this->_scene->addItem( this->_photo );
	this->setScene( this->_scene );

	this->setTransformationAnchor( QGraphicsView::ViewportAnchor::AnchorUnderMouse );
	this->setResizeAnchor( QGraphicsView::ViewportAnchor::AnchorUnderMouse );
	this->setVerticalScrollBarPolicy( Qt::ScrollBarPolicy::ScrollBarAlwaysOff );
	this->setHorizontalScrollBarPolicy( Qt::ScrollBarPolicy::ScrollBarAlwaysOff );
	this->setBackgroundBrush( QBrush( QColor( 30, 30, 30 ) ) );
	this->setFrameShape( QFrame::NoFrame );

	image_buffer = nullptr;
	//this->temp_color_convert.create( 1080, 1980, CV_8UC3 );
}

ScalableGraphicsView::~ScalableGraphicsView()
{

}

void ScalableGraphicsView::resizeEvent( QResizeEvent* event )
{
	//if( this->_photo != nullptr && this->_zoom == 0 )
	if( !this->_photo.isNull() && this->_zoom == 0 )
		fitInView( 0, 0, this->_photo.size().width(), this->_photo.size().height(), Qt::KeepAspectRatio );

	QGraphicsView::resizeEvent( event );
}

void ScalableGraphicsView::setPicture( const std::string & file_name )
{
	this->_zoom = 0;
	if( this->_photo_handle != nullptr )
	{
		this->_scene->removeItem( this->_photo_handle );
		this->_photo_handle = nullptr;
	}

	this->_photo.load( file_name.c_str() );

	if( !this->_photo.isNull() )
	{
		this->_photo_handle = this->_scene->addPixmap( this->_photo );
		this->setDragMode( QGraphicsView::DragMode::ScrollHandDrag );
		this->fitImageInView();
	}
	else
	{
		this->setDragMode( QGraphicsView::DragMode::NoDrag );
	}
}
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"

void ScalableGraphicsView::setPicture( const pcv::RGBA_UChar_Image & image )
{
	//this->_stored_image = image;
	//image.copyTo( this->_stored_image );
	if( this->_photo_handle != nullptr )
	{
		this->_scene->removeItem( this->_photo_handle );
		delete this->_photo_handle;
		this->_photo_handle = nullptr;
	}

	if( image.data == nullptr )
		return;

	//Mat & dest = this->_stored_image;
	//cvtColor( image, dest, CV_BGRA2RGB );
	//Mat & dest = this->temp_color_convert;
	Mat dest = image;
	//Mat dest;// = image.clone();
	//cv::cuda::GpuMat gpu_image;
	//gpu_image.upload