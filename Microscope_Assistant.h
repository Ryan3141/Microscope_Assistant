#pragma once

#include <QtWidgets/QMainWindow>
#include <QLine>
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
	void keyReleaseEvent( QKeyEvent* event );

private:
	Ui::Microscope_AssistantClass ui;

	QSettings* settings;
	Device_Communicator* my_devices;
	Camera_Interface* camera;
	Live_Stitcher* stitcher;

	QLine line1;
	QLine line2;
	QGraphicsLineItem* line_drawn1 = nullptr;
	QGraphicsLineItem* line_drawn2 = nullptr;

	void Main_Loop();
	void Start_Looking_For_Connections( QWidget *parent );
	void Draw_Total_Image( const cv::Mat & image );
	void Save_Overall_Image() const;

	pcv::BGRA_UChar_Image current_overall_image;

	qint64 old_time;
	qint64 time_sum = 0;
	int sum_counter = 0;
	const int pixels_per_mm = 3562;
};
