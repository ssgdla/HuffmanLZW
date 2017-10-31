import SocketServer
import time
import functions
from PIL import Image

class MyTcpServer(SocketServer.BaseRequestHandler):
    def recvfile(self, filename):
        print "Start receiving file!"
        f = open(filename, 'wb')
        self.request.send('ready')
        while True:
            data = self.request.recv(4096)
            if data == 'EOF':
                print 'receive file successfully'
                break
            f.write(data)
        f.close()
    
    def sendfile(self, filename):
        print "start sending file!"
        self.request.send('ready')
        time.sleep(1)
        f = open(filename, 'rb')
        while True:
            data = f.read(4096)
            if not data:
                break
            self.request.send(data)
        f.close()
        time.sleep(1)
        self.request.send('EOF')
        print "send file successfully"
        
    def handle(self):
        print "get connection from :",self.client_address
        while True:
            try:
                data = self.request.recv(4096)
                print "get data:", data  
                if not data:
                    print "break the connection!"
                    break               
                else:
                    action, filename = data.split()
                    if action == "put":
                        LZWresult = functions.compressImage(filename)
                        functions.writeFile(LZWresult)
                        self.sendfile('someFile.bin')
                    elif action == 'get':
                        self.recvfile(filename)
                        LZWresult = ''
                        with open('someFile.bin', 'rb') as f:
                            LZWresult = f.read()
                        f.close()
                        data = functions.decompressImage(LZWresult)
                        new_im = Image.fromarray(data)
                        new_im.convert('RGB').save('out.jpg')
                    else:
                        print "get error!"
                        continue
            except Exception,e:
                print "get error at:",e
                                            
                                        
if __name__ == "__main__":
    host = ''
    port = 60002
    s = SocketServer.ThreadingTCPServer((host,port), MyTcpServer)
    s.serve_forever()
        