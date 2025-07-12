import socket

ip = "192.168.143.88"  # Replace with your bot's IP
port = 80

s = socket.socket()
s.connect((ip, port))

while True:
    cmd = input("Enter command (w/a/s/d/x): ")
    if cmd in ['w', 'a', 's', 'd', 'x']:
        s.send(cmd.encode())
    else:
        print("Invalid command")
