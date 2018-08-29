#pragma once

#include <QObject>
#include <QHostAddress>
#include <QMap>
#include <QtSql>

class QTcpServer;
class QUdpSocket;
class QPlainTextEdit;
class QTcpSocket;

struct Device
{
	QTcpSocket* pSocket;
	QString name;
	QStringList header_info;
	QString listener;
	QString table_name;
	QSqlQuery sql_insert_command;
	QStringList value_binds;
	QString raw_data_stream;

	//bool operator==(const Device & d) const;
	bool Build_SQL_Insert_From_First_Message( const QString & device_type, const QStringList & first_message, const QMap<QString, QString> & identifier_to_table );
	bool Insert_New_Data( const QStringList & data );
};

class Device_Communicator : public QObject
{
	Q_OBJECT

public:
	Device_Communicator( QObject *parent, const QMap<QString, QString> & identifier_to_table, const QHostAddress & listener_address, unsigned short port );
	~Device_Communicator();
	void Poll_LocalIPs_For_Devices( QString ip_range );

	void Update_Listener_Info( const QMap<QString, QString> & identifier_to_table );
signals:
	void Reply_Recieved( QString reply, Device & device );
	void Command_Sent( const QString & command, const Device & device );
	void Device_Disconnected( QString device_ip );

public slots:
	void Send_Command( const QByteArray & command, Device & device );
	void Send_Command( const QByteArray & command );

private:
	bool Listen_For_Replies( QHostAddress ip_to_listen_on );
	void Handle_New_Connection();
	QStringList Convert_IP_Range_To_List( const QString & ip_range );
	QStringList Recursive_Convert_IP_Range_To_List( const QString & ip_range, int position );

	QMap<QString, Device> active_connections;
	unsigned short port_for_ping;
	unsigned short next_port_to_use;

	QTcpServer* tcp_server;
	QUdpSocket* udpSocket;

	QMap<QString, QString> device_identifier_to_sql_table;
};
