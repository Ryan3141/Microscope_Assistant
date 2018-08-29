#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_Microscope_Assistant.h"

class QSettings;
class Device_Communicator;

class Microscope_Assistant : public QMainWindow
{
	Q_OBJECT

public:
	Microscope_Assistant(QWidget *parent = Q_NULLPTR);

private:
	Ui::Microscope_AssistantClass ui;

	QSettings* settings;
	Device_Communicator* my_devices;

	void Start_Looking_For_Connections( QWidget *parent );
};
