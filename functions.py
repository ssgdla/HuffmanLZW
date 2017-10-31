import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import math

#The DCT matrix
DCTmatrix = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        if i==0:
            DCTmatrix[i,j] = 1.0/np.sqrt(8.0)
        else:
            DCTmatrix[i,j] = (np.sqrt(0.25))*math.cos(((2.0*j+1)*i*np.pi)/(2.0*8))  

def getQuantizationMatrix(switch):
    #Quantization matrix with quality level of 10
    if switch == 10:    
        Q10 = np.array([[80,60,50,80,120,200,255,255],
                        [55,60,70,95,130,255,255,255],
                        [70,65,80,120,200,255,255,255],
                        [70,85,110,145,255,255,255,255],
                        [90,110,185,255,255,255,255,255],
                        [120,175,255,255,255,255,255,255],
                        [245,255,255,255,255,255,255,255],
                        [255,255,255,255,255,255,255,255]])
        return Q10
    elif switch == 50:
        #Quantization matrix with quality level of 50
        Q50 = np.array([[16,11,10,16,24,40,51,61],
                        [12,12,14,19,26,58,60,55],
                        [14,13,16,24,40,57,69,56],
                        [14,17,22,29,51,87,80,62],
                        [18,22,37,56,68,109,103,77],
                        [24,35,55,64,81,104,113,92],
                        [49,64,78,87,103,121,120,101],
                        [72,92,95,98,112,100,103,99]])
        return Q50
    elif switch == 90:
        #Quantization matrix with quality level of 90
        Q90 = np.array([[3,2,2,3,5,8,10,12],
                        [2,2,3,4,5,12,12,11],
                        [3,3,3,5,8,11,14,11],
                        [3,3,4,6,10,17,16,12],
                        [4,4,7,11,14,22,21,15],
                        [5,7,11,13,16,12,23,18],
                        [10,13,16,17,21,24,24,21],
                        [14,18,19,20,22,20,20,20]])
        return Q90
    else:
        print("error switch")

#Help functions for compression include:
#findZlist(matrix)
#RLCmethod(Zlist)   
#findVLI(Coef)
#findMedianList(rlc)
#getCode(medianList) 

#Return matrix after zigzag scan, ending 0s are replaced by 'EOB'
def findZlist(matrix):
    result = []
    result.append(matrix[0,0])
    r = 0
    c = 0
    for i in range(1,64):
        while i<64 and (r-1)>=0 and (c+1)<8:
            result.append(matrix[r-1,c+1])
            r = r-1
            c = c+1
        if i<64 and (c+1)<8: 
            result.append(matrix[r,c+1])
            c = c + 1
        elif i<64 and (r+1)<8:
            result.append(matrix[r+1,c])
            r =r + 1
        while i<64 and (r+1)<8 and (c-1)>=0:
            result.append(matrix[r+1,c-1])
            r = r + 1
            c = c - 1
        if i<64 and (r+1)<8:
            result.append(matrix[r+1,c])
            r= r + 1
        elif i<64 and (c+1)<8:
            result.append(matrix[r,c+1])
            c = c+ 1
    for i in range(63,-1,-1):
        if result[i] == 0 and i != 0:
            result.pop(i)
        elif result[i] == 0 and i == 0:
            result.append('EOB')
        elif result[i] != 0 and i == 63:
            break
        else:
            result.append('EOB')
            break
    return result

#Run Lengh Coding
def RLCmethod(Zlist):
    result = []
    result.append(Zlist[0])
    i = 1
    count = 0
    size = len(Zlist)
    while i<size and Zlist[i] != 'EOB':
        if Zlist[i] == 0:
            if(count == 15):
                result.append(15)
                result.append(0)
                count = 0
            count = count + 1
        else:
            result.append(count)
            result.append(Zlist[i])
            count = 0
        i = i + 1       
    result.append('EOB')
    return result

#Find category and real value based on VLI table for given coefficient 
def findVLI(Coef):
    Coef = int(Coef)
    category = 0
    RealValue = ""
    if Coef == 0:
        return category,RealValue
    real = 0
    for i in range(1,16): #Category 1 to 15
        if Coef < np.power(2,i) and Coef > -1*np.power(2,i):
            category = i
            if Coef<0:
                real = Coef - (-np.power(2,i)+1)
            else:
                real = Coef
            break
    binaryValue = bin(real)    
    realValue = binaryValue[2:]
    NumZeros = category - len(realValue)
    for i in range(NumZeros):
        j = 0
        realValue = str(j) + realValue
    return category,realValue       

#Find median list
def findMedianList(rlc):
    result = []
    dc = rlc[0]
    dc_category, dc_RealValue = findVLI(dc)
    result.append(dc_category) # Add DC category and DC difference into median list
    result.append(dc)
    i = 1
    while(i<len(rlc)):
        if(rlc[i] != 'EOB'):
            result.append(rlc[i])
            i = i + 1
            ACcategory, ACrealValue = findVLI(rlc[i])
            result.append(ACcategory)
            result.append(rlc[i])
            i = i + 1
        else:
            break
    result.append(0) #For EOB and (0,0)
    result.append(0)   
    return result  
  
#Table for DC coefficient difference
DCcode = np.array(['000','010','011','100','101','110','1110','11110','111110','1111110','11111110','111111110'])

#Table for AC coefficient    
ACcode = np.array([['1010','00','01','100','1011','11010','1111000','11111000','1111110110','1111111110000010','1111111110000011'],
                   ['X','1100','11011','1111001','111110110','11111110110','1111111110000100', \
                   '1111111110000101','1111111110000110','1111111110000111','1111111110001000'],
                   ['X','11100','11111001','1111110111','111111110100','1111111110001001','1111111110001010','1111111110001011', \
                    '1111111110001100','1111111110001101','1111111110001110'],
                   ['X','111010','111110111','111111110101','1111111110001111','1111111110010000','1111111110010001', \
                    '1111111110010010','1111111110010011','1111111110010100','1111111110010101'],
                   ['X','111011','1111111000','1111111110010110','1111111110010111','1111111110011000','1111111110011001', \
                    '1111111110011010','1111111110011011','1111111110011100','1111111110011101'],
                   ['X','1111010','11111110111','1111111110011110','1111111110011111','1111111110100000','1111111110100001', \
                    '1111111110100010','1111111110100011','1111111110100100','1111111110100101'],
                   ['X','1111011','111111110110','1111111110100110','1111111110100111','1111111110101000','1111111110101001', \
                    '1111111110101010','1111111110101011','1111111110101100','1111111110101101'],
                   ['X','11111010','111111110111','1111111110101110','1111111110101111','1111111110110000','1111111110110001', \
                    '1111111110110010','1111111110110011','1111111110110100','1111111110110101'],
                   ['X','111111000','111111111000000','1111111110110110','1111111110110111','1111111110111000','1111111110111001', \
                    '1111111110111010','1111111110111011','1111111110111100','1111111110111101'],
                   ['X','111111001','1111111110111110','1111111110111111','1111111111000000','1111111111000001','1111111111000010', \
                    '1111111111000011','1111111111000100','1111111111000101','1111111111000110'], 
                   ['X','111111010','1111111111000111','1111111111001000','1111111111001001','1111111111001010','1111111111001011', \
                    '1111111111001100','1111111111001101','1111111111001110','1111111111001111'],
                   ['X','1111111001','1111111111010000','1111111111010001','1111111111010010','1111111111010011','1111111111010100', \
                    '1111111111010101','1111111111010110','1111111111010111','1111111111011000'], 
                   ['X','1111111010','1111111111011001','1111111111011010','1111111111011011','1111111111011100','1111111111011101', \
                    '1111111111011110','1111111111011111','1111111111100000','1111111111100001'],
                   ['X','11111111000','1111111111100010','1111111111100011','1111111111100100','1111111111100101','1111111111100110', \
                    '1111111111100111','1111111111101000','1111111111101001','1111111111101010'],
                   ['X','1111111111101011','1111111111101100','1111111111101101','1111111111101110','1111111111101111','1111111111110000', \
                    '1111111111110001','1111111111110010','1111111111110011','1111111111110100'],
                   ['11111111001','1111111111110101','1111111111110110','1111111111110111','1111111111111000','1111111111111001', \
                    '1111111111111010','1111111111111011','1111111111111100','1111111111111101','1111111111111110']])

#Get the final code to be stored    
def getCode(medianList):
    result = ""
    result = result + DCcode[medianList[0]]
    c,r = findVLI(medianList[1])
    result = result + r
    i = 2
    length = len(medianList)
    while i<length-2:
        ac = ACcode[medianList[i], medianList[i+1]]
        result = result + ac
        i = i+2
        cc,rr = findVLI(medianList[i])
        result = result + rr
        i = i+1
    result = result + ACcode[0,0]
    return result

#Help functions for decompression include:
#getValueFromCategoryAndRealValue(RealValue, Category)
#reconstructMedian(code)
#reconstructRLC(medianCode)
#recontructZList(RLClist)
#reconstructZMatrix(z)

#Get value based on category and code stored
def getValueFromCategoryAndRealValue(RealValue, Category): #RealValue is a string, category is a int
    binary = '0b' + RealValue
    integer = eval(binary)
    if integer >= np.power(2,(Category-1)):
        return integer
    else:
        integer = integer - (np.power(2,Category) - 1)
        return integer

#Reconstruct Median code based on compressed code
def reconstructMedian(code):
    size = 0 #Size of compressed code for current submatrix
    i = 0
    result = []
    for dcCategory in range(len(DCcode)):
        if DCcode[dcCategory] == code[0:len(DCcode[dcCategory])]:
            result.append(dcCategory)
            size = size + len(DCcode[dcCategory])
            break
    i = len(DCcode[dcCategory]) #Next bit to be considered
    dcRealValue = code[i:(i+dcCategory)]
    if dcCategory != 0:    
        dcValue = getValueFromCategoryAndRealValue(dcRealValue, dcCategory)
        result.append(dcValue)
        size = size + len(dcRealValue)
    else:
        result.append(0)
        
    i = i + dcCategory
    while i<len(code)-3:
        stopwhile = False
        for ii in range(16):
            stopiiLoop = False
            for jj in range(11):
                tempLen = len(ACcode[ii,jj])
                if ACcode[ii,jj] == code[i:(i+tempLen)]:
                    result.append(ii)
                    result.append(jj)
                    size = size + len(ACcode[ii,jj])
                    stopiiLoop = True
                    i = i+tempLen
                    if ii== 15 and jj==0:
                        result.append(0)
                        break
                    elif ii==0 and jj==0:
                        stopwhile = True
                        break
                    else:
                        ACrealValue = code[i:(i+jj)]
                        acValue = getValueFromCategoryAndRealValue(ACrealValue, jj)
                        result.append(acValue)
                        size = size + jj
                        i = i + jj
                        break
            if stopiiLoop == True:
                break
        if stopwhile == True:
            break
    return result,size

#Reconstruct RLC code based on median code
def reconstructRLC(medianCode):
    result = []
    result.append(medianCode[1])
    length = len(medianCode)
    for i in range(2,length,3):
        if (i+2)<length:
            result.append(medianCode[i])
            result.append(medianCode[i+2])
    result.append('EOB')
    return result            

#Reconstructed Zigzag list based on RLC list    
def recontructZList(RLClist):
    result = []
    result.append(RLClist[0])
    length = len(RLClist)
    for ri in range(1,length,2):
        if ri == length-1:
            break
        if RLClist[ri]>0:
            for rj in range(RLClist[ri]):
                result.append(0)
        result.append(RLClist[ri+1])
        if (ri+2) == length-1:
            break
    if len(result)<64:
        added = 64-len(result)
        for ri in range(added):
            result.append(0)
    return result

#Reconstruct ZigZag matrix based on zigzag list
def reconstructZMatrix(z):
    result = np.array([[z[0],z[1],z[5],z[6],z[14],z[15],z[27],z[28]],
                       [z[2],z[4],z[7],z[13],z[16],z[26],z[29],z[42]],
                       [z[3],z[8],z[12],z[17],z[25],z[30],z[41],z[43]],
                       [z[9],z[11],z[18],z[24],z[31],z[40],z[44],z[53]],
                       [z[10],z[19],z[23],z[32],z[39],z[45],z[52],z[54]],
                       [z[20],z[22],z[33],z[38],z[46],z[51],z[55],z[60]],
                       [z[21],z[34],z[37],z[47],z[50],z[56],z[59],z[61]],
                       [z[35],z[36],z[48],z[49],z[57],z[58],z[62],z[63]]
                       ])
    return result    

def LZWCompression(uncompressed):
    """Compress a string to a list of output symbols."""
 
    # Build the dictionary.
    dict_size = 256
    dictionary = dict((chr(i), chr(i)) for i in range(dict_size))
 
    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
 
    # Output the code for w.
    if w:
        result.append(dictionary[w])
    return result
 
 
def LZWDecompression(compressed):
    """Decompress a list of output ks to a string."""

 
    dict_size = 256
    dictionary = dict((chr(i), chr(i)) for i in range(dict_size))

 
    w = result = compressed.pop(0)
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        result += entry
 
        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
 
        w = entry
    return result

def compressImage(imageName): #imageName is the string name of the image file
    #All original images have same size 1024*1024
    #ImageList = ["portrait.jpg", "indoor.jpg", "outdoor.png"] 
    #ImageList = ['portrait.jpg']
    #ImageList = ['indoor.jpg']
    #ImageList = ['outdoor.png']
    #for nImage in ImageList:
    CompressedCode = ""
    Q = getQuantizationMatrix(90) #Parameter could be chosen in 10, 50, 90 represents different quality level
    image = Image.open(imageName).convert('L')
    img = np.array(image)
    imgC = np.zeros((1024,1024)) # Full image after quantization
    dc = 0 #Initiate DC value
    #For every 8*8 block perform DCT transform       
    for i in range(128): #128 blocks in row direction
        for j in range(128): #128 blocks in column direction
            matrix = np.zeros((8,8))
            for ii in range(8): #Get the block matrix
                matrix[ii,:] = img[i*8+ii, j*8:(j+1)*8]
            #Because DCT is designed to work on -128 to 127, the original block is "level off"
            #by subtracting 128 from each entry.
            matrix = matrix - 128   
            #Perform DCT transform
            matrix = np.dot(np.dot(DCTmatrix,matrix),np.linalg.inv(DCTmatrix))
            Cmatrix = np.zeros((8,8))
            for ii in range(8):
                for jj in range(8):
                    Cmatrix[ii,jj] = round(matrix[ii,jj]/Q[ii,jj])
            
            #The following is the compression process
            Zlist = findZlist(Cmatrix)
            tempDC = Zlist[0]
            Zlist[0] = Zlist[0] - dc #DPCM of DC coefficient(find value difference of current DC and previous DC)     
            dc = tempDC  #reset DC value for calculation of next matrix
            RLClist = RLCmethod(Zlist) #List after run length coding
            MedianList = findMedianList(RLClist) #Find the median list ready for final coding
            stringCode = getCode(MedianList)
            CompressedCode = CompressedCode + stringCode #Add new code into compressed code
      #     for ii in range(8):
      #         imgC[i*8+ii, j*8:(j+1)*8] = Cmatrix[ii,:]        
    #Calculate the compression ratio
    originalSize = 1024*1024*8
    print(len(CompressedCode))
#    for i in range(1024):
#        for j in range(1024):
#            originalSize = originalSize + len(bin(int(imgC[i,j]))) - 2
    print("Compression ratio: " + str(len(CompressedCode)/(originalSize*1.0)))

    LZWresult = LZWCompression(CompressedCode)
    print("Comparison")
    print(8*len(LZWresult))
    print(len(CompressedCode))
    return LZWresult

def writeFile(LZWresult):
    with open('someFile.bin', 'wb') as f:
        f.write(LZWresult)
    f.close()

def decompressImage(CompressedCode):
    CompressedCode = LZWDecompression(CompressedCode)
    #The following code is for the decompression process
    Q = getQuantizationMatrix(90) #Parameter could be chosen in 10, 50, 90 represents different quality level
    imgC = np.zeros((1024,1024))
    DC = 0
    for i in range(128):
        for j in range(128):
            medianList,size = reconstructMedian(CompressedCode)
            RLClist = reconstructRLC(medianList)
            Zlist = recontructZList(RLClist)
            zMatrix = reconstructZMatrix(Zlist)
            zMatrix[0,0] = zMatrix[0,0] + DC
            DC = zMatrix[0,0]       
            for ii in range(8):
                imgC[i*8+ii, j*8:(j+1)*8] = zMatrix[ii,:]
            CompressedCode = CompressedCode[size:]    

    
    # Image reconstruction using imgC                   
    for i in range(128):
        for j in range(128):
            matrix = np.zeros((8,8))
            for ii in range(8): #Get the block matrix
                matrix[ii,:] = imgC[i*8+ii, j*8:(j+1)*8]
            for ii in range(8):
                for jj in range(8):
                    matrix[ii,jj] = Q[ii,jj]*matrix[ii,jj]
            matrix = np.dot(np.dot(np.linalg.inv(DCTmatrix), matrix), DCTmatrix)
            for ii in range(8):
                for jj in range(8):
                    matrix[ii,jj] = round(matrix[ii,jj])
            matrix = matrix + 128
            for ii in range(8):
                imgC[i*8+ii, j*8:(j+1)*8] = matrix[ii,:]
                
    #Show the image after reconstruction            
    #new_im = Image.fromarray(imgC)
    return imgC #return image array
#    new_im.show()
#    new_im.convert('RGB').save('out.jpg') #Save compressed images
    