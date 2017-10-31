import socket
import time
import functions
from PIL import Image

ip = '192.168.1.128'
port = 60002
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
def recvfile(filename):
    print "server ready, now client rece file~~"
    f = open(filename, 'wb')
    while True:
        data = s.recv(4096)
        if data == 'EOF':
            print "recv file success!"
            break
        f.write(data)
    f.close()
def sendfile(filename):
    print "server ready, now client sending file~~"
    f = open(filename, 'rb')
    while True:
        data = f.read(4096)
        if not data:
            break
        s.sendall(data)
    f.close()
    time.sleep(1)
    s.sendall('EOF')
    print "send file success!"
                                
def confirm(s, client_command):
    s.send(client_command)
    data = s.recv(4096)
    if data == 'ready':
        return True
                                
try:
    s.connect((ip,port))
    while 1:
        client_command = raw_input(">>")
        if not client_command:
            continue
                                    
        action, filename = client_command.split()
        if action == 'put':
            if confirm(s, client_command):
                LZWresult = functions.compressImage(filename)
                functions.writeFile(LZWresult)
                sendfile(filename)
            else:
                print "server get error!"
        elif action == 'get':
            if confirm(s, client_command):
                recvfile(filename)
                LZWresult = ''
                with open('someFile.bin', 'rb') as f:
                    LZWresult = f.read()
                f.close()
                data = functions.decompressImage(LZWresult)
                new_im = Image.fromarray(data)
                new_im.convert('RGB').save('out.jpg')
            else:
                print "server get error!"
        else:
            print "command error!"
except socket.error,e:
    print "get error as",e
finally:
    s.close()