import sys
import numpy as np
from itertools import combinations

def calc_error(samples):
    error = 0
    for i in range(6):
        error += ((2*x[x.index(samples[i][0])] - 1) -samples[i][1]) **2
    return error

samples = []
x = [i for i in range(-5,7)]
mean, sigma = 0, np.sqrt(2)
gaussian_noise = np.random.normal(mean,sigma,12)
for i in range(12):
    samples.append([x[i],2*x[i] - 1 + gaussian_noise[i]])

comb = list(combinations(samples,6))

min_error = sys.maxsize
ans_6 = []
for i in range(len(comb)):
    data = comb[i]
    A = []
    b = []
    for j in range(6):
        A.append([data[j][0],1])
        b.append(data[j][1])
    A = np.array(A)
    b = np.array(b)

    AT = np.transpose(A)
    AT_A_inv = np.linalg.inv(np.matmul(AT,A))
    AT_b = np.matmul(AT,b)
    res = np.matmul(AT_A_inv,AT_b)

    error = calc_error(comb[i])
    if error < min_error:
        min_error = error
        ans_6 = res
print(ans_6)

A = []
b = []
for j in range(12):
    A.append([x[j],1])
    b.append(samples[j][1])
A = np.array(A)
b = np.array(b)

AT = np.transpose(A)
AT_A_inv = np.linalg.inv(np.matmul(AT,A))
AT_b = np.matmul(AT,b)
res = np.matmul(AT_A_inv,AT_b)
print(res)
        


