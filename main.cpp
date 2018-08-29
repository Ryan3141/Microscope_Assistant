#include "Microscope_Assistant.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	Microscope_Assistant w;
	w.show();
	return a.exec();
}
