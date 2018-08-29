#ifndef SCALABLEGRAPHICSVIEW_H
#define SCALABLEGRAPHICSVIEW_H

#include <string>

#include <QGraphicsView>

#include <opencv2/core.hpp>

class ScalableGraphicsView : public QGraphicsView
{
	Q_OBJECT

public:
	ScalableGraphicsView( QWidget *parent );
	~ScalableGraphicsView();
	virtual void resizeEvent( QResizeEvent* event );

	void setPicture( const std::string & file_name );
	void setPicture( const cv::Mat & image );
	void fitImageInView();

	int _zoom;
	QGraphicsScene* _scene;
	QPixmap* _photo;
	QGraphicsPixmapItem* _photo_handle;

private:
	virtual void mouseMoveEvent( QMouseEvent * event );
	virtual void mousePressEvent( QMouseEvent* event );
	//virtual void mouseReleaseEvent( QMouseEvent* event );
	virtual void wheelEvent( QWheelEvent* event );
	cv::Mat _stored_image;

	//bool _pan;
	//int _panStartX, _panStartY;
	//qreal _scale;

signals:
	void rightClicked( int image_x, int image_y );
};

#endif // SCALABLEGRAPHICSVIEW_H
