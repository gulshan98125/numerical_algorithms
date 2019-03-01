import numpy as np
import random
datasetwa = [random.gauss(1000,0.00000000001) for i in range(3000)]
def var1(x):
	res = 0
	for i in range(len(x)):
		res += x[i]**2 
	return res/np.float64(len(x)) - (mean(x)**2)

def var2(x):
	mean_val = mean(x)
	res = 0
	for i in x:
		res += (i-mean_val)**2
	return res/np.float64(len(x))

def mean(x):
	res = 0
	for i in x:
		res+=i
	return res/np.float64(len(x))

def data1():
	return [9.99999997, 9.99999966, 9.99999972]

def kahanSum(x):
	sumwa = x[0]
	c = np.float64(0)
	for i in range(1,len(x)):
		y = x[i] - c
		t = sumwa + y
		c = (t - sumwa) - y
		sumwa = t
	return sumwa

def meanKahan(x):
	return kahanSum(x)/np.float64(len(x))

def var3(x):
	mean_val = meanKahan(x)
	arr2 = [(i-mean_val)**2 for i in x]
	return kahanSum(arr2)/np.float64(len(x))

def data2():
	return datasetwa

dataset = [9.99999997, 9.99999966, 9.99999972]

print(var1(data2()))
print(var2(data2()))
print(var3(data2()))