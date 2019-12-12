#include "Microscope_Assistant.h"

#include <QSettings>

#include "Device_Communicator.h"
#include "Camera_Interface.h"
#include "ScalableGraphicsView.h"
#include "Live_Stitcher.h"

#include <opencv2/core.hpp>
Q_DECLARE_METATYPE( cv::Mat )

Microscope_Assistant::Microscope_Assistant( QWidget *parent )
	: QMainWindow(parent)
{
	//test_main();
	settings = new QSettings( "configuration.ini", QSettings::IniFormat, this );

	//Start_Looking_For_Connections( parent );
	ui.setupUi(this);

	{
		QThread* thread = new QThread;
		this->camera = new Camera_Interface;
		this->camera->moveToThread( thread );
		connect( thread, &QThread::started, this->camera, &Camera_Interface::Start_Thread );
		connect( this->camera, &Camera_Interface::Work_Finished, thread, &QThread::quit );
		//automatically delete thread and task object when work is done:
		connect( thread, &QThread::finished, this->camera, &QObject::deleteLater );
		connect( thread, &QThread::finished, thread, &QObject::deleteLater );
		connect( ui.zDown_pushButton, &QPushButton::clicked, this->camera, &Camera_Interface::Take_Image );
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
		thread->start();
	}

	//connect( stitcher, &Live_Stitcher::Display_Image, ui.total_graphicsView, qOverload<const cv::Mat &>( &ScalableGraphicsView::setPicture ) );
	connect( stitcher, &Live_Stitcher::Display_Image, this, [this]( const cv::Mat & image )
	{ // This connect requires cv::Mat to be declared as a metatype with Q_DECLARE_METATYPE( cv::Mat )
		ui.total_graphicsView->setPicture( image );
	}, Qt::QueuedConnection );
}

void Microscope_Assistant::Main_Loop()
{
	cv::Mat image = this->camera->Get_Image();
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
