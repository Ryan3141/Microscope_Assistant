#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_Microscope_Assistant.h"

class QSettings;
class Device_Communicator;
class Camera_Interface;
class Live_Stitcher;

class Microscope_Assistant : public QMainWindow
{
	Q_OBJECT

public:
	Microscope_Assistant(QWidget *parent = Q_NULLPTR);

private:
	Ui::Microscope_AssistantClass ui;

	QSettings* settings;
	Device_Communicator* my_devices;
	Camera_Interface* camera;
	Live_Stitcher* stitcher;

	void Main_Loop();
	void Start_Looking_For_Connections( QWidget *parent );
	void Draw_Total_Image( const cv::Mat & image );

	qint64 old_time;
	qint64 time_sum = 0;
	int sum_counter = 0;
};
