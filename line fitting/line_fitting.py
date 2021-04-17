import numpy as np
from itertools import combinations

# 8 Sample points
points = [
    [-2.9,35.4],[-2.1,19.7],[-0.9,5.7],[1.1,2.1],
    [0.1,1.2],[1.9,8.7],[3.1,25.7],[4.0,41.5]
]
for k in range(28):
    # Pick 6 Samples out of 8 Sample points
    comb = list(combinations(list(range(8)),6))
    comb2 = list(combinations(list(range(28)),2))
    first_idx, second_idx = comb2[k]
    first_case = []
    second_case = []
    for i in range(6):
        first_case.append(points[comb[first_idx][i]])
        second_case.append(points[comb[second_idx][i]])

    arr1 = []
    arr2 = []
    b1 = []
    b2 = []

    for i in range(6):
        arr1.append([pow(first_case[i][0],2),first_case[i][0],1])
        b1.append(first_case[i][1])
        arr2.append([pow(second_case[i][0],2),second_case[i][0],1])
        b2.append(second_case[i][1])

    A1 = np.array(arr1)
    A2 = np.array(arr2)

    A1T = np.transpose(A1)
    A2T = np.transpose(A2)

    A1TA1_inv = np.linalg.inv(np.matmul(A1T,A1))
    A2TA2_inv = np.linalg.inv(np.matmul(A2T,A2))

    A1Tb1 = np.matmul(A1T,b1)
    A2Tb2 = np.matmul(A2T,b2)

    res1 = np.matmul(A1TA1_inv,A1Tb1)
    res2 = np.matmul(A2TA2_inv,A2Tb2)

    print("(x,y): ",end=' ')
    for j in range(6):
        print("(",A1[j][1],b1[j],")",end=' ')
    print()
    print("case" + str(k) + '[1] '+ str(res1))
    print("(x,y): ",end=' ')
    for j in range(6):
        print("(",A2[j][1],b2[j],")",end=' ')
    print()
    print("case" + str(k) + '[2] '+ str(res2))

