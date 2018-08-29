#include "Device_Communicator.h"

#include <vector>
#include <assert.h>

#include <QTcpServer>
#include <QUdpSocket>
#include <QTcpSocket>
#include <QTimer>
#include <QMap>
#include <memory>

//static QString Sanitize_SQL( const QString & raw_string )
//{ // Got regex from https://stackoverflow.com/questions/9651582/sanitize-table-column-name-in-dynamic-sql-in-net-prevent-sql-injection-attack
//	if( raw_string.contains( ';' ) )
//		return QString();
//
//	QRegularExpression re( R"(^[\p{L}{\p{Nd}}$#_][\p{L}{\p{Nd}}@$#_]*$)" );
//	QRegularExpressionMatch match = re.match( raw_string );
//	bool hasMatch = match.hasMatch();
//
//	if( hasMatch )
//		return raw_string;
//	else
//		return QString();
//}
//
//bool Device::operator==(const Device & d) const
//{
//	return (this->pSocket == d.pSocket);
//}

//bool Device::Build_SQL_Insert_From_First_Message( const QString & device_type, const QStringList & first_message, const QMap<QString, QString> & identifier_to_table )
//{
//	if( !identifier_to_table.contains( device_type ) )
//		return false;
//
//	QSqlQuery query;
//	QStringList headers;
//	for( int i = 0; i < this->header_info.size(); i++ )
//	{
//		headers.append( Sanitize_SQL( this->header_info[ i ] ) );
//		value_binds.append( ":v" + QString::number( i ) );
//	}
//	this->sql_insert_command.prepare( "INSERT INTO " + Sanitize_SQL( identifier_to_table[ this->listener ] )
//									  + " (location," + headers.join( ',' ) + ",time) "
//									  "VALUES (:location," + value_binds.join( ',' ) + ",now())" );
//
//	this->sql_insert_command.bindValue( ":location", this->name );
//	return true;
//}
//
bool Device::Insert_New_Data( const QStringList & data )
{
	if( data.size() < this->value_binds.size() )
		return false;

	for( int i = 0; i < this->value_binds.size(); i++ )
	{
		if( data[i].toLower() == "null" )
			this->sql_insert_command.bindValue( value_binds[ i ], QVariant( QVariant::String ) );
		else
			this->sql_insert_command.bindValue( value_binds[ i ], data[ i ] );
	}
	auto debug1 = this->sql_insert_command.exec();
	QString another = this->sql_insert_command.executedQuery();
	auto debug4 = this->sql_insert_command.lastError().text();
	return true;
}

Device_Communicator::Device_Communicator( QObject *parent, const QMap<QString, QString> & identifier_to_table, const QHostAddress & listener_address, unsigned short port )
	: QObject( parent ), device_identifier_to_sql_table( identifier_to_table )
{
	qRegisterMetaType<QAbstractSocket::SocketError>();
	port_for_ping = port;
	tcp_server = new QTcpServer( this );
	udpSocket = new QUdpSocket( this );
	//Listen_For_Replies( QHostAddress::LocalHost, port );
	if( !Listen_For_Replies( listener_address ) )
		throw QString( "Failed to find local network " + listener_address.toString() );
}

void Device_Communicator::Poll_LocalIPs_For_Devices( QString ip_range )
{
	QStringList potential_ip_addresses = Convert_IP_Range_To_List( ip_range );
	for( QString key : active_connections.keys() )
	{ // Don't ask again if we are already connected
		QStringList split_by_colon = key.split( ':' );
		split_by_colon.removeLast();
		potential_ip_addresses.removeOne( split_by_colon.join( ':' ) );
	}
	for( QMap<QString, QString>::const_iterator one_identifier = this->device_identifier_to_sql_table.begin(); one_identifier != this->device_identifier_to_sql_table.end(); ++one_identifier )
	{
		QByteArray test_message = one_identifier.key().toUtf8();// "Crystal Monitor";// "Cold Plate Listener";
		for( QString ip_address : potential_ip_addresses )
		{
			udpSocket->writeDatagram( test_message.data(), test_message.size(), QHostAddress( ip_address ), port_for_ping );
		}
	}
}

Device_Communicator::~Device_Communicator()
{
}

void Device_Communicator::Update_Listener_Info( const QMap<QString, QString> & identifier_to_table )
{
	this->device_identifier_to_sql_table = identifier_to_table;
}

bool Device_Communicator::Listen_For_Replies( QHostAddress ip_to_listen_on )
{
	bool result = tcp_server->listen( ip_to_listen_on, this->port_for_ping );
	if( !result )
		return false;

	//connect( tcp_server, &QTcpServer::, this, &Device_Communicator::Handle_New_Connection );
	connect( tcp_server, &QTcpServer::newConnection, this, &Device_Communicator::Handle_New_Connection );

	// Ping connections to test for them disconnecting unexpectedly
	QTimer *timer = new QTimer( this );
	connect( timer, &QTimer::timeout, [this] { this->Send_Command( QByteArray( "PING" ) ); } );
	timer->start( 2000 );

	return true;
}

void Device_Communicator::Handle_New_Connection()
{
	//active_connections.push_back( Device{} );
	
	QTcpSocket* new_pSocket = tcp_server->nextPendingConnection();
	QString peer_ip = new_pSocket->peerAddress().toString();
	QString peer_port = QString::number( new_pSocket->peerPort() );

	QString peer_identifier = peer_ip + ":" + peer_port;
	Device & new_connected_device = active_connections[ peer_identifier ];
	new_connected_device.pSocket = new_pSocket;
	qInfo() << QDateTime::currentDateTime().toString() + QString( ": Response from %1:%2" ).arg( peer_ip ).arg( peer_port );
	// Tell TCP socket to timeout if unexpectedly disconnected
	new_connected_device.pSocket->setSocketOption( QAbstractSocket::KeepAliveOption, 1 );

	//auto conn = std::make_shared<QMetaObject::Connection>();
	connect( new_connected_device.pSocket, &QTcpSocket::disconnected,
			 [this, peer_identifier]
	{
		//disconnect( *conn );
		Device & new_connected_device = this->active_connections[ peer_identifier ];
		emit Device_Disconnected( new_connected_device.pSocket->peerAddress().toString() );
		this->active_connections.erase( this->active_connections.find( peer_identifier ) );// std::remove( this->active_connections.begin(), this->active_connections.end(), new_connected_device ), this->active_connections.end() );
		qInfo() << QDateTime::currentDateTime().toString() + QString( ": Disconnected from %1" ).arg( peer_identifier );
	} );

	connect( new_connected_device.pSocket, &QTcpSocket::readyRead, [this, peer_identifier]
	{
		Device & new_connected_device = this->active_connections[ peer_identifier ];
		QByteArray data = new_connected_device.pSocket->readAll();
		//new_connected_device.pSocket->write( "I'm Valid\n" );
		//log->appendPlainText( data + '\n' );
		new_connected_device.raw_data_stream += QString::fromUtf8( data );
		QStringList split_by_line = new_connected_device.raw_data_stream.split( '\n' );
		new_connected_device.raw_data_stream = split_by_line.last();
		split_by_line.removeLast();

		for( QString one_line : split_by_line )
		{
			emit Reply_Recieved( one_line, new_connected_device );
		}
	} );
}
void Device_Communicator::Send_Command( const QByteArray & command, Device & device )
{
	device.pSocket->write( command + '\n' );

	emit Command_Sent( command, device );
}

void Device_Communicator::Send_Command( const QByteArray & command )
{
	for( Device & d : this->active_connections )
		Send_Command( command, d );
}

QStringList Device_Communicator::Convert_IP_Range_To_List( const QString & ip_range )
{
	return Recursive_Convert_IP_Range_To_List( ip_range, 0 );
}

QStringList Device_Communicator::Recursive_Convert_IP_Range_To_List( const QString & ip_range, int position )
{
	if( position == 4 )
		return QStringList{ ip_range };

	QStringList split_ip = ip_range.split( "." );
	QStringList range = split_ip[ position ].split( "-" );

	if( range.size() == 1 )
	{
		assert( range[ 0 ].toInt() > 0 && range[ 0 ].toInt() < 255 );
		return Recursive_Convert_IP_Range_To_List( ip_range, position + 1 );
	}
	else if( range.size() == 2 )
	{
		assert( range[ 0 ].toInt() < range[ 1 ].toInt() );
		assert( range[ 0 ].toInt() > 0 && range[ 0 ].toInt() < 255 );
		assert( range[ 1 ].toInt() > 0 && range[ 1 ].toInt() < 255 );
		QStringList big_list;
		for( int i = range[ 0 ].toInt(); i <= range[ 1 ].toInt(); i++ )
		{
			split_ip[ position ] = QString::number( i );
			QString ip_with_this_i = split_ip.join( '.' );
			big_list += Recursive_Convert_IP_Range_To_List( ip_with_this_i, position + 1 );
		}

		return big_list;
	}
	return{};
}
