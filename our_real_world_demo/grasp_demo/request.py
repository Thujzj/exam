from socket import *

client_socket = socket(AF_INET, SOCK_DGRAM)
server_host_port = ("10.6.8.62",8889)
# client_socket.sendto("function_1,饼干".encode("utf-8"), server_host_port)
client_socket.sendto("function_1,饮料".encode("utf-8"), server_host_port)