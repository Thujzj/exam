from socket import *

client_socket = socket(AF_INET, SOCK_DGRAM)
server_host_port = ("127.0.0.1",8889)

client_socket.sendto("hello".encode("utf-8"), server_host_port)
client_socket.close()