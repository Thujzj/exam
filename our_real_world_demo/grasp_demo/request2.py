from socket import *

client_socket = socket(AF_INET, SOCK_DGRAM)
server_host_port = ("10.6.8.62",8889)
client_socket.sendto("function_2,盒子".encode("utf-8"), server_host_port)