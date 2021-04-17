import numpy as np
import cv2
import sys
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

def is_1(I):
    if I[0] == [0, 16] and I[1] == [2, 16] and I[2] == [2, 18]:
        print("this picture is 1!")
        return True
    else: return False
def is_2(I):
    if I[0] == [28, 0] and I[1] == [28, 1] and I[2] ==[28, 4]:
        print("this picture is 2!")
        return True
    else: return False
def is_3(I):
    if I[0] == [15, 0] and I[1] == [15,3] and (I[2] ==[20,0] or I[2] ==[21,0]):
        print("this picture is 3!")
        return True
    else: return False
def is_4(I):
    if ([23,5] in I) and ([23,6] in I):
        print("this picture is 4!")
        return True
    else: return False
def is_5(I):
    if ([3,0] in I) and([4,0] in I) and ([13,19] in I):
        print("this picture is 5!")
        return True
    else: return False
def is_6(I):
    if (I[0][0] == 20) and (I[1][0] == 20 or I[1][0] == 21) and (I[2][0] == 20 or I[2][0] == 21):
        print("this picture is 6!")
        return True
    else: return False
def is_7(I):
    if ([21,8] in I) and([23,6] in I) and ([25,5] in I):
        print("this picture is 7!")
        return True
    else: return False
def is_8(I):
    if ([7,0] in I) and([9,0] in I) and ([22,0] in I) and ([24,0] in I):
        print("this picture is 8!")
        return True
    else: return False
def is_9(I):
    if ([22,0] in I) and([27,0] in I) and ([30,2] in I):
        print("this picture is 9!")
        return True
    else: return False
def is_10(I):
    if ([15,0] in I) and([20,8] in I) and ([24,9] in I):
        print("this picture is 10!")
        return True
    else: return False
def is_11(I):
    if ([23,9] in I) and([25,7] in I) and ([27,5] in I) and ([29,3] in I):
        print("this picture is 11!")
        return True
    else: return False
def is_12(I):
    if (I[9][0] == 31) and (I[8][0] == 31 or I[8][0] == 30 or I[8][0] == 29):
        print("this picture is 12!")
        return True
    else: return False
def is_13(I):
    if ([16,0] in I) and([21,0] in I) and ([24,0] in I) and ([26,0] in I):
        print("this picture is 13!")
        return True
    else: return False
def is_14(I):
    if ([30,0] in I) and([30,2] in I):
        print("this picture is 14!")
        return True
    else: return False
def is_15(I):
    if ([25,2] == I[0] or [25,2] == I[1]) and ([31,0] in I) and ([31,1] in I):
        print("this picture is 15!")
        return True
    else: return False
def is_16(I):
    if (([12,0] == I[0] or [4,0] == I[0]) and [14,0] == I[1] and [15,0] == I[2]) or \
        ([1,0] == I[0] and [2,0] == I[1] and [8,0] == I[2]) or \
        ([5,0] == I[0] and [8,0] == I[1] and [9,0] == I[2] and [11,0] == I[3]):
        print("this picture is 16!")
        return True
    else: return False
def is_17(I):
    if ([9,0] in I) and ([29,1] == I[9]):
        print("this picture is 17!")
        return True
    else: return False
def is_18(I):
    if ([17,0] in I) and ([22,0] in I) and ([24,0] in I) and ([25,0] in I) and ([27,0] in I) and ([28,0] in I):
        print("this picture is 18!")
        return True
    else: return False
def is_19(I):
    if ([23,0] in I) and ([26,0] in I) and ([26,3] in I) and ([26,5] in I) and ([29,0] in I):
        print("this picture is 19!")
        return True
    else: return False
def is_20(I):
    if ([19,0] == I[0] and [24,0] == I[1]) and ([27,0] in I) and ([28,0] in I):
        print("this picture is 20!")
        return True
    else: return False 
INF = sys.maxsize
img_path = 'fabric'
files = [ f for f in listdir(img_path) if isfile(join(img_path,f)) ]

dominant = [] # 10개의 큰 index 위치

for i in range(len(files)):
    img = cv2.imread(join(img_path,files[i]),0)
    if i==0 or i==1:
        img = cv2.resize(img,dsize=(160,160),interpolation=cv2.INTER_AREA)
    elif i==2:
        img = cv2.resize(img,dsize=(320,320),interpolation=cv2.INTER_AREA)  
    elif i==4 or i==12:
        img = cv2.resize(img,dsize=(130,130),interpolation=cv2.INTER_AREA)
    elif i==6 or i==10:
        img = cv2.resize(img,dsize=(200,200),interpolation=cv2.INTER_AREA)
    elif i==18:
        img = cv2.resize(img,dsize=(360,360),interpolation=cv2.INTER_AREA)
    # 6(480*640) 10(240*640) 15(426*640) 17(478*640) 20(300*600)
    elif i==5 or i==9 or i==14 or i==16 or i==19:
        img = cv2.resize(img,dsize=(0,0),fx=0.4,fy=0.4,interpolation=cv2.INTER_AREA)
    # 8(360*640)
    elif i==7:
        img = cv2.resize(img,dsize=(0,0),fx=0.7,fy=0.7,interpolation=cv2.INTER_AREA)

    rows,cols = img.shape
    r = rows // 64
    c = cols // 64

    coefficients = []

    for j in range(r):
        for k in range(c):
            _img = img[j*64:(j*64)+64,k*64:(k*64)+64]
            f = np.fft.fft2(_img)
            fshift = np.fft.fftshift(f)
            m_spectrum = 20*np.log(np.abs(fshift))

            quadrant = m_spectrum[:32,32:64]

            tmp = np.ravel(quadrant)
            tmp = np.sort(tmp)[::-1]
            cut = tmp[9]

            I = []
            x = 0
            for p in range(32):
                for q in range(32):
                    if quadrant[p][q] >= cut:
                        if x == 10: break
                        I.append([p,q])
                        x += 1
                if x==10: break

            coefficients.append(I)
    dominant.append(coefficients)
    #print("--------------------------------------------------------------------------")

#exit()

for i in range(len(files)):
    img = cv2.imread(join(img_path,files[i]),0)
    if i==0 or i==1:
        img = cv2.resize(img,dsize=(160,160),interpolation=cv2.INTER_AREA)
    elif i==2:
        img = cv2.resize(img,dsize=(320,320),interpolation=cv2.INTER_AREA)  
    elif i==4 or i==12:
        img = cv2.resize(img,dsize=(130,130),interpolation=cv2.INTER_AREA)
    elif i==6 or i==10:
        img = cv2.resize(img,dsize=(200,200),interpolation=cv2.INTER_AREA)
    elif i==5 or i==9 or i==14 or i==16 or i==19:
        img = cv2.resize(img,dsize=(0,0),fx=0.4,fy=0.4,interpolation=cv2.INTER_AREA)
    elif i==7:
        img = cv2.resize(img,dsize=(0,0),fx=0.7,fy=0.7,interpolation=cv2.INTER_AREA)
    elif i==18:
        img = cv2.resize(img,dsize=(360,360),interpolation=cv2.INTER_AREA)
    img = img[:64,:64]

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    m_spectrum = 20*np.log(np.abs(fshift))

    block = m_spectrum[:32,32:64]

    tmp = np.ravel(block)
    tmp = np.sort(tmp)[::-1]
    cut = tmp[9]

    I = []
    x = 0
    for p in range(32):
        for q in range(32):
            if block[p][q] >= cut:
                if x == 10: break
                I.append([p,q])
                x += 1
        if x==10: break
    
    while True:
        if is_1(I) or is_2(I): break
        elif is_3(I) or is_5(I) or is_7(I) or is_8(I) or is_10(I) or is_11(I) or is_13(I) or \
            is_18(I) or is_19(I) or is_20(I): break
        elif is_4(I) or is_9(I) or is_14(I) or is_15(I) or is_17(I): break
        elif is_16(I): break
        elif is_6(I): break
        elif is_12(I): break
            

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(m_spectrum, cmap = 'gray') 
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()