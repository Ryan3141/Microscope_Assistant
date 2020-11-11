#include "Microscope_Assistant.h"

#include <iostream>
#include <QSettings>
#include <QFileDialog>
#include <QKeyEvent>
#include <QGraphicsLineItem>

#include "Device_Communicator.h"
#include "Camera_Interface.h"
#include "ScalableGraphicsView.h"
#include "Live_Stitcher.h"

#include <opencv2/core.hpp>
Q_DECLARE_METATYPE( cv::Mat )
Q_DECLARE_METATYPE( pcv::RGBA_UChar_Image )


static void Keep_Inside( cv::Rect & area, const cv::Mat & image_to_keep_inside )
{
	int cols = image_to_keep_inside.cols;
	int rows = image_to_keep_inside.rows;
	int original_x = area.x;
	int original_y = area.y;
	area.x = std::min( cols - 1, std::max( 0, original_x ) );
	area.width = std::min( cols - area.x, std::min( area.width + original_x, area.width ) );
	area.y = std::min( rows - 1, std::max( 0, original_y ) );
	area.height = std::min( rows - area.y, std::min( area.height + original_y, area.height ) );
}

static void Keep_Inside( cv::Rect & area, const QPixmap & image_to_stay_inside )
{
	int cols = image_to_stay_inside.size().width();
	int rows = image_to_stay_inside.size().height();
	int original_x = area.x;
	int original_y = area.y;
	area.x = std::min( cols - 1, std::max( 0, original_x ) );
	area.width = std::min( cols - area.x, std::min( area.width + original_x, area.width ) );
	area.y = std::min( rows - 1, std::max( 0, original_y ) );
	area.height = std::min( rows - area.y, std::min( area.height + original_y, area.height ) );
}

static cv::Rect Get_Search_Rect( QPoint search_location, const cv::Mat & image_to_stay_inside, int size_ratio )
{
	int cols = image_to_stay_inside.cols;
	int rows = image_to_stay_inside.rows;
	cv::Rect section_to_use( search_location.x() - cols / (2 * size_ratio), search_location.y() - rows / (2 * size_ratio), cols / size_ratio, rows / size_ratio );
	Keep_Inside( section_to_use, image_to_stay_inside );
	return section_to_use;
}

static cv::Rect Get_Search_Rect( QPoint search_location, const QPixmap & image_to_stay_inside, int size_ratio )
{
	int cols = image_to_stay_inside.size().width();
	int rows = image_to_stay_inside.size().height();
	cv::Rect section_to_use( search_location.x() - cols / (2 * size_ratio), search_location.y() - rows / (2 * size_ratio), cols / size_ratio, rows / size_ratio );
	Keep_Inside( section_to_use, image_to_stay_inside );
	return section_to_use;
}

Microscope_Assistant::Microscope_Assistant( QWidget *parent )
	: QMainWindow( parent )
{
	//test_main();
	settings = new QSettings( "configuration.ini", QSettings::IniFormat, this );

	//Start_Looking_For_Connections( parent );
	ui.setupUi( this );

	{
		QThread* thread = new QThread;
		this->camera = new Camera_Interface;
		this->camera->moveToThread( thread );
		connect( thread, &QThread::started, this->camera, &Camera_Interface::Start_Thread );
		connect( this->camera, &Camera_Interface::Work_Finished, thread, &QThread::quit );
		//automatically delete thread and task object when work is done:
		connect( thread, &QThread::finished, this->camera, &QObject::deleteLater );
		connect( thread, &QThread::finished, thread, &QObject::deleteLater );
		//connect( ui.addImage_pushButton, &QPushButton::clicked, this->camera, &Camera_Interface::Take_Image );
		thread->start();
	}

	QTimer* main_loop_timer = new QTimer( this );
	connect( main_loop_timer, &QTimer::timeout, this, &Microscope_Assistant::Main_Loop );
	main_loop_timer->start( 1000 / 60 );

	{
		QThread* thread = new QThread;
		this->stitcher = new Live_Stitcher( this->camera, nullptr );
		this->stitcher->moveToThread( thread );
		connect( thread, &QThread::started, this->stitcher, &Live_Stitcher::Start_Thread );
		connect( this->stitcher, &Live_Stitcher::Work_Finished, thread, &QThread::quit );
		//automatically delete thread and task object when work is done:
		connect( thread, &QThread::finished, this->stitcher, &QObject::deleteLater );
		connect( thread, &QThread::finished, thread, &QObject::deleteLater );
		connect( ui.addImage_pushButton, &QPushButton::clicked, this->stitcher, &Live_Stitcher::Stitch_Image_And_Start_New );
		connect( ui.resetStitching_pushButton, &QPushButton::clicked, this->stitcher, &Live_Stitcher::Reset_Stitching );
		connect( ui.selectFolder_pushButton, &QPushButton::clicked, [=]
		{
			QString folder = QFileDialog::getExistingDirectory( this, tr( "Folder To Save Files In" ), QString() );
			if( folder != "" )
				ui.saveLocation_lineEdit->setText( folder );
		} );
		connect( ui.saveImage_pushButton, &QPushButton::clicked, this, &Microscope_Assistant::Save_Overall_Image );
		//connect( ui.saveImage_pushButton, &QPushButton::clicked, [=]
		//{
		//	QString folder = this->ui.saveLocation_lineEdit->text();
		//	QString sample = this->ui.sampleName_lineEdit->text();
		//	QString number = this->ui.imageNumber_lineEdit->text();
		//	QString file_name = QString( "%1 (%2).jpg" ).arg( sample ).arg( number );
		//	this->ui.imageNumber_lineEdit->setText( QString::number( number.toInt() + 1 ) );
		//	QFileInfo file( folder, file_name );
		//	QMetaObject::invokeMethod( this->stitcher, [=]
		//	{
		//		this->stitcher->Save_Image_And_Start_Over( file );
		//	}, Qt::QueuedConnection );
		//} );
		thread->start();
	}
	//connect( stitcher, &Live_Stitcher::Display_Image, this, [this]( const cv::Mat & image )
	//{ // This connect requires cv::Mat to be declared as a metatype with Q_DECLARE_METATYPE( cv::Mat )
	//	QFileInfo file_name = QFileDialog::getSaveFileName( this, tr( "Save Data" ), QString(), tr( "CSV File (*.csv)" ) );//;; JPG File (*.jpg);; BMP File (*.bmp);; PDF File (*.pdf)" ) );
	//}
	//connect( stitcher, &Live_Stitcher::Display_Image, ui.total_graphicsView, qOverload<const cv::Mat &>( &ScalableGraphicsView::setPicture ) );
	connect( stitcher, &Live_Stitcher::Display_Image, this, [this]( const pcv::BGRA_UChar_Image & image )
	{ // This connect requires cv::Mat to be declared as a metatype with Q_DECLARE_METATYPE( cv::Mat )
		ui.total_graphicsView->setPicture( image );
		this->current_overall_image = image;
	}, Qt::QueuedConnection );
	connect( stitcher, &Live_Stitcher::Display_Debug_Image, this, [this]( const pcv::BGRA_UChar_Image & image )
	{ // This connect requires cv::Mat to be declared as a metatype with Q_DECLARE_METATYPE( cv::Mat )
		ui.debug_graphicsView->setPicture( image );
	}, Qt::QueuedConnection );

	connect( ui.live_graphicsView, &ScalableGraphicsView::rightClicked,
			 [this]( int x, int y )
	{
		line1.setP1( line1.p2() );
		line1.setP2( QPoint( x, y ) );
		const QPoint delta = ((line1.p2() - line1.p1()).x() > 0) ? (line1.p2() - line1.p1()) : (line1.p1() - line1.p2());
		ui.lineLength1_lineEdit->setText( QString::number( std::sqrt( delta.x() * delta.x() + delta.y() * delta.y() ), 'g', 6 ) );
		ui.lineAngle1_lineEdit->setText( QString::number( -(360.0 / 2 / 3.1415926535) * std::atan( double( delta.y() ) / delta.x() ), 'g', 6 ) );
		if( this->line_drawn1 == nullptr )
		{
			//ui.total_graphicsView->_scene->removeItem( this->line_drawn );
			//QBrush fillBrush( Qt::transparent );
			QPen outlinePen( Qt::red );
			outlinePen.setWidth( 2 );
			this->line_drawn1 = ui.live_graphicsView->_scene->addLine( line1, outlinePen );
			this->line_drawn1->setZValue( 1.0 );
		}
		else
			this->line_drawn1->setLine( line1 );
	} );

	connect( ui.total_graphicsView, &ScalableGraphicsView::rightClicked,
			 [this]( int x, int y )
	{
		line2.setP1( line2.p2() );
		line2.setP2( QPoint( x, y ) );
		const QPoint delta = ((line2.p2() - line2.p1()).x() > 0) ? (line2.p2() - line2.p1()) : (line2.p1() - line2.p2());
		ui.lineLength2_lineEdit->setText( QString::number( std::sqrt( delta.x() * delta.x() + delta.y() * delta.y() ), 'g', 6 ) );
		ui.lineAngle2_lineEdit->setText( QString::number( -(360.0 / 2 / 3.1415926535) * std::atan( double( delta.y() ) / delta.x() ), 'g', 6 ) );
		if( this->line_drawn2 == nullptr )
		{
			//ui.total_graphicsView->_scene->removeItem( this->line_drawn );
			//QBrush fillBrush( Qt::transparent );
			QPen outlinePen( Qt::red );
			outlinePen.setWidth( 2 );
			this->line_drawn2 = ui.total_graphicsView->_scene->addLine( line2, outlinePen );
			this->line_drawn2->setZValue( 1.0 );
		}
		else
			this->line_drawn2->setLine( line2 );
	} );

}

void Microscope_Assistant::Save_Overall_Image() const
{
	QString folder = this->ui.saveLocation_lineEdit->text();
	QString sample = this->ui.sampleName_lineEdit->text();
	QString number = this->ui.imageNumber_lineEdit->text();
	QString file_name = QString( "%1 (%2).jpg" ).arg( sample ).arg( number );
	this->ui.imageNumber_lineEdit->setText( QString::number( number.toInt() + 1 ) );
	QFileInfo file( folder, file_name );
	QMetaObject::invokeMethod( this->stitcher, [=]
	{
		this->stitcher->Save_Image_And_Start_Over( file );
	}, Qt::QueuedConnection );
}

void Microscope_Assistant::Main_Loop()
{
	pcv::RGBA_UChar_Image image = this->camera->Get_Image();
	ui.live_graphicsView->setPicture( image );

	{
		qint64 new_time = QDateTime::currentMSecsSinceEpoch();
		time_sum += new_time - old_time;
		sum_counter++;
		old_time = new_time;
		if( sum_counter >= 30 )
		{
			double fps = 1000.0 * sum_counter / time_sum;
			//qDebug() << "Framerate to display: " << fps << "\n";
			time_sum = 0;
			sum_counter = 0;
		}
	}
	//QThread::msleep( 1000 / 60 );
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

void Microscope_Assistant::keyReleaseEvent( QKeyEvent* event )
{
	if( event->key() == Qt::Key::Key_Space )
	{
		QMetaObject::invokeMethod( this->stitcher, [=]()
		{
			this->stitcher->Stitch_Image_And_Start_New();
		}, Qt::QueuedConnection );
	}
	if( event->key() == Qt::Key::Key_Enter || event->key() == Qt::Key::Key_Return )
	{
		this->Save_Overall_Image();
		//QString folder = this->ui.saveLocation_lineEdit->text();
		//QString sample = this->ui.sampleName_lineEdit->text();
		//QString number = this->ui.imageNumber_lineEdit->text();
		//QString file_name = QString( "%1 (%2).jpg" ).arg( sample ).arg( number );
		//this->ui.imageNumber_lineEdit->setText( QString::number( number.toInt() + 1 ) );
		//QFileInfo file( folder, file_name );
		//QMetaObject::invokeMethod( this->stitcher, [=]()
		//{
		//	this->stitcher->Save_Image_And_Start_Over( file );
		//}, Qt::QueuedConnection );
	}
}