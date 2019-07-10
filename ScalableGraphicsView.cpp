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

void ScalableGraphicsView::setPicture( const Mat & image )
{
	if( image.data == nullptr )
		return;
	//this->_stored_image = image;
	//image.copyTo( this->_stored_image );
	if( this->_photo_handle != nullptr )
	{
		this->_scene->removeItem( this->_photo_handle );
		delete this->_photo_handle;
		this->_photo_handle = nullptr;
	}

	//Mat & dest = this->_stored_image;
	//cvtColor( image, dest, CV_BGRA2RGB );
	//Mat & dest = this->temp_color_convert;
	Mat dest = image;
	//Mat dest;// = image.clone();
	//cv::cuda::GpuMat gpu_image;
	//gpu_image.upload( image );
	////cv::cuda::GpuMat bgr_image;
	////bgr_image.create( 1080, 1980, CV_8UC3 );
	//cv::cuda::cvtColor( gpu_image, gpu_image, CV_BGR2RGB );
	//gpu_image.download( dest );
	//cvtColor( image, dest, CV_BGR2RGBA );
	this->_photo = QPixmap(); // Get rid of old pixmap which referenced memory in _stored_image
	this->_stored_image = image; // _stored_image holds a reference count to keep the data alive for the pixmap
	//uchar* rawPtr = new uchar[ dest.step * dest.rows ];
	//if( image_buffer == nullptr )
	//	image_buffer = new uchar[ dest.step * dest.rows ];

	//memcpy( image_buffer, (uchar*)dest.data, dest.step * dest.rows );
	QImage q_image( this->_stored_image.data, dest.cols, dest.rows, dest.step, QImage::Format_RGB32 );
	//QImage q_image( rawPtr, dest.cols, dest.rows, int( dest.step ), QImage::Format_RGB32, []( void* allocated ) { delete[] allocated; }, rawPtr );
	//QImage q_image( (uchar*)dest.data, dest.cols, dest.rows, int( dest.step ), QImage::Format_ARGB32, []( void* allocated ) { delete[] allocated; }, rawPtr );
	//QImage q_image( (uchar*)dest.data, dest.cols, dest.rows, dest.step, QImage::Format_RGBA8888 );
	//int debug2 = 1;
	//QPixmap::fromImage( q_image );
	//new QPixmap( QPixmap::fromImage( q_image ) );
	//this->_photo = new QPixmap( QPixmap::fromImage( q_image ) );
	this->_photo = QPixmap::fromImage( std::move(q_image) );
	//this->_photo.loadFromData( (uchar*)dest.data, );
	auto debug = this->_photo.isNull();

	if( !this->_photo.isNull() )
	{
		needs_initial_setup = false;
		this->_photo_handle = this->_scene->addPixmap( this->_photo );
		this->setDragMode( QGraphicsView::DragMode::ScrollHandDrag );
		setSceneRect( this->_scene->itemsBoundingRect() );
		if( needs_initial_setup )
		{
			this->_zoom = 0;
			this->fitImageInView();
		}
	}
	else
	{
		this->setDragMode( QGraphicsView::DragMode::NoDrag );
	}
}

void ScalableGraphicsView::fitImageInView()
{
	QRectF rect( this->_photo.rect() );
	if( rect.isNull() )
		return;

	QRectF to_identity = this->transform().mapRect( QRectF( 0, 0, 1, 1 ) );
	this->scale( 1 / to_identity.width(), 1 / to_identity.height() );
	QRectF view_rect( this->viewport()->rect() );
	QRectF scene_rect( this->transform().mapRect( rect ) );
	double factor = min( view_rect.width() / scene_rect.width(),
						 view_rect.height() / scene_rect.height() );
	this->scale( factor, factor );
	this->centerOn( rect.center() );
	this->_zoom = 0;
}

void ScalableGraphicsView::mousePressEvent( QMouseEvent *event )
{
	if( event->button() == Qt::RightButton )
	{
		//_pan = true;
		//_panStartX = event->x();
		//_panStartY = event->y();
		//setCursor( Qt::ClosedHandCursor );
		setCursor( Qt::CrossCursor );

		// get scene coords from the view coord
		QPointF scenePt = mapToScene( event->pos() );
		//emit this->rightClicked( scenePt.x(), scenePt.y() );

		// get the item that was clicked on
		QGraphicsItem* the_item = this->_scene->itemAt( scenePt, transform() );

		if( the_item )
		{
			// get the scene pos in the item's local coordinate space
			QPointF localPt = the_item->mapFromScene( scenePt );
			emit this->rightClicked( localPt.x(), localPt.y() );
		}

		event->accept();
		return;
	}
	QGraphicsView::mousePressEvent( event );
	//event->ignore();
}
//
//void ScalableGraphicsView::mouseReleaseEvent( QMouseEvent *event )
//{
//	if( event->button() == Qt::RightButton )
//	{
//		_pan = false;
//		setCursor( Qt::ArrowCursor );
//		event->accept();
//		return;
//	}
//	event->ignore();
//}
//
void ScalableGraphicsView::mouseMoveEvent( QMouseEvent *event )
{
	QGraphicsView::mouseMoveEvent( event );
	viewport()->setCursor( Qt::CrossCursor );
	//	if( _pan )
	//	{
	//		//horizontalScrollBar()->setValue( horizontalScrollBar()->value() - (event->x() - _panStartX) );
	//		//verticalScrollBar()->setValue( verticalScrollBar()->value() - (event->y() - _panStartY) );
	//		translate( -(event->x() - _panStartX), -(event->y() - _panStartY) );
	//		_panStartX = event->x();
	//		_panStartY = event->y();
	//		event->accept();
	//		return;
	//	}
	//	event->ignore();
}

void ScalableGraphicsView::wheelEvent( QWheelEvent* event )
{
	int debug = event->delta();
	auto change2 = event->angleDelta();

	if( this->_photo.isNull() )
		return;

	double change = 1.0;
	if( event->angleDelta().y() > 0 )
	{
		change = 1.25;
		this->_zoom++;
	}
	else
	{
		change = 0.8;
		this->_zoom--;
	}

	if( this->_zoom > 0 )
		this->scale( change, change );
	else if( this->_zoom == 0 )
		this->fitImageInView();
	else
	{
		this->fitImageInView();
		this->_zoom = 0;
	}

	event->accept();
}
