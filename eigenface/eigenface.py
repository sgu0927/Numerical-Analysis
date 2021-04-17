import sys
from os import listdir
from os.path import isfile, join
from copy import deepcopy
import numpy as np
import cv2
import itertools

img_path = 'images'
files = [ f for f in listdir(img_path) if isfile(join(img_path,f)) ]
A = np.empty((len(files),1024),dtype=object) # row vectors dtype=object?

for i in range(len(files)): # 3000
    img = cv2.imread(join(img_path,files[i]),0) # 0 해줘야 흑백
    resize_img = cv2.resize(img,(32,32))
    tmp = []
    for j in range(32): # tmp_len 1024
        for k in range(32):
            #astype cast 안해주면 RuntimeWarning: overflow encountered in ubyte_scalars
            tmp.append(resize_img[j][k].astype(float))
    print("A[%d] is done" %i)
    A[i] = tmp

_M = np.mean(A)
M = [_M] * 1024
M = np.array(M)

# covariance matrix
# True -> row가 변수, col이 관측치
# False -> row가 관측치, col이 변수
# C.shape :  (1024, 1024)
C = np.cov(A.astype(float), rowvar=False)

# #s = singular values 1차원 1024개 나옴
# V- row space basis U - col space basis
# orthonormal 값 return

U,s,Vh =  np.linalg.svd(C, full_matrices = True)
# V is unitary, Vt 안에는 row vector로 row space basis 들어있음
# 정렬도 되어있습니다.


name = ["Abdullah_Gul","Adrien_Brody","Ai_Sugiyama","Al_Gore","Al_Sharpton",
    "Alastair_Campbell","Albert_Costa","Alejandro_Toledo","Ali_Naimi","Allyson_Felix"]
test_path = 'test'
files = [ f for f in listdir(test_path) if isfile(join(test_path,f)) ]
res = []
for i in range(10):
    c = [[0 for _ in range(45)] for _ in range(5)]
    for j in range(5):
        img = cv2.imread(join(test_path,files[i*5 + j]) ,0)
        resize_img = cv2.resize(img,(32,32))

        test = []
        for k in range(32):
            for l in range(32):
                test.append(resize_img[k][l].astype(float))
        test = np.array(test)
        test -= M
        for k in range(45):
            c[j][k] = np.dot(test,Vh[k])
        res.append(c[j])  


# 각 case간 차이 제곱 정보 
comb = list(itertools.combinations(range(5),2))
comb2 = list(itertools.combinations(range(10),2))
for i in range(len(comb2)):
    c,d = comb2[i]
    min_max = []
    for k in range(len(comb)):
        a,b = comb[k]
        dist = []
        for l in range(45):
            dist.append((res[a+5*c][l] - res[b+5*d][l])**2)
        min_max.append(sum(dist))
    print("case %d - %d" %(c,d))
    print("min: ",min(min_max))
    print("max: ",max(min_max))

# 자기 자신 중 가장 variance 작은 coefficient 
for i in range(10):
    tmp = [0 for _ in range(45)]
    for j in range(len(comb)):
        a,b = comb[j]
        for k in range(45):
            tmp[k] += (res[5*i+a][k] - res[5*i+b][k]) ** 2
    print("min distance at %d" %(i+1),tmp.index(min(tmp))+1)