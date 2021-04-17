import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

def getC(u,v):
    if u == 0 and v == 0: return (0.5)
    elif (u==0 and v>0) or (u>0 and v==0): return (1/np.sqrt(2))
    else: return 1

def dct(Color):
    F = [[0 for _ in range(16)] for _ in range(16)]
    for v in range(16):
        for u in range(16):
            _sum = 0
            for x in range(16):
                for y in range(16):
                    _sum += float(Color[x][y]) * np.cos((v*np.pi) * (2*x + 1) / (2 * float(N))) * \
                        np.cos((u*np.pi) * (2*y + 1) / (2 * float(N)))
            F[v][u] = _sum * getC(u,v) * (1/8)

    F = np.array(F)
    tmp = np.ravel(list(map(abs,F)))
    tmp = np.sort(tmp)[::-1]
    cut = tmp[15]

    cnt = 0
    for i in range(16):
        for j in range(16):
            if cnt == 16:
                F[i][j] = 0
            elif abs(F[i][j]) < cut: 
                F[i][j] = 0
            else:
                cnt += 1

    return F

def idct(F):
    S = [[0 for _ in range(16)] for _ in range(16)]
    for x in range(16):
        for y in range(16):
            _sum = 0
            for v in range(16):
                for u in range(16):
                    _sum += getC(u,v) * float(F[v][u]) * np.cos((v*np.pi) * (2*x + 1) / (2 * float(N))) * \
                        np.cos((u*np.pi) * (2*y + 1) / (2 * float(N)))
            S[x][y] = _sum * (1/8)

    return S

N = 16

# 1(256*256) 2(720*960) 3(720*960)
img_path = 'test2'
files = [ f for f in listdir(img_path) if isfile(join(img_path,f)) ]

max_j = 256
max_k = 256

for i in range(3):
    img = cv2.imread(join(img_path,files[i]),1)
    (B, G, R) = cv2.split(img)

    if i>0 :
        max_j = 720
        max_k = 960

    for j in range(0,max_j,16):
        for k in range(0,max_k,16):
            tmp = dct(R[j:j+16,k:k+16])
            tmp = idct(tmp)
            for p in range(16):
                for q in range(16):
                    R[j+p][k+q] = tmp[p][q]
            tmp = dct(G[j:j+16,k:k+16])
            tmp = idct(tmp)
            for p in range(16):
                for q in range(16):
                    G[j+p][k+q] = tmp[p][q]
            tmp = dct(B[j:j+16,k:k+16])
            tmp = idct(tmp)
            for p in range(16):
                for q in range(16):
                    B[j+p][k+q] = tmp[p][q]
            print("%d is done!!" % k)
        print("[j]%d is done!!" % j)
    for j in range(max_j):
        for k in range(max_k):
            if R[j][k] < 0: R[j][k] = 0
            elif R[j][k] >255: R[j][k] = 255
            if G[j][k] < 0: G[j][k] = 0
            elif G[j][k] >255: G[j][k] = 255
            if B[j][k] < 0: B[j][k] = 0
            elif B[j][k] >255: B[j][k] = 255
    merged = cv2.merge([B,G,R])

    cv2.imshow("Merged", merged)
    cv2.waitKey(0)              

    cv2.destroyAllWindows()
    
