import numpy as np
from numpy import linalg as LA
import makeNetwork as mn
import matplotlib.pyplot as plt
import math
def applyA(network,v):
	#maps v to A.v in O(m) time
	(m, tuples) = network
	Av = np.zeros(m)
	Av = Av.tolist()
	for t in tuples:
		i,j = t[0],t[1]
		Rij = t[2]
		Av[i] += (v[i]*(1./Rij) - v[j]*(1./Rij))
		Av[j] += (v[j]*(1./Rij) - v[i]*(1./Rij))
	Av[0] += v[0]
	Av[m-1] += v[m-1]

	#A formation is completed in O(m) time 
	return np.array(Av)

def cg(Afun, b, tolerance):
	x = np.zeros(len(b))
	r = b
	p = b
	counter = 0
	residual_norms = []
	b_norm = LA.norm(b,2)
	while(LA.norm(r,2)/b_norm > tolerance and (counter<len(b))):
		Ap_n_minus1 = Afun(p) # Afun takes 2d vector
		rT_r_old = np.dot(r.T,r) # previous value of (r transpose)*r
		alpha = rT_r_old/np.dot(p.T, Ap_n_minus1);
		x = x + (alpha*p)
		# print(Ap_n_minus1)
		residual_norms.append(math.log(LA.norm(r,2)/b_norm))
		r = r - (alpha*Ap_n_minus1);
		beta = np.dot(r.T,r)/rT_r_old
		p = r + (beta*p)
		counter+=1
	return x,residual_norms

def getB(network):
	m,tuples = network
	b = np.zeros(m)
	b[0] = -1
	return b

def getA_from_network(network):
	(m, tuples) = network
	A = np.zeros((m,m))
	for t in tuples:
		i,j = t[0],t[1]
		Rij = t[2]
		A[i][j], A[j][i] = -1./Rij,-1./Rij
		A[i][i] += 1./Rij
		A[j][j] += 1./Rij
	A[0][0] += 1
	A[m-1][m-1] += 1

	#A formation is completed in O(m) time 
	return A

def getDiag(network):
	return getA_from_network(network).diagonal()


def pcg(Afun, b, d, tolerance):
	M_inverse = np.zeros((len(d),len(d)))
	#inverse of diagonal matrix is fractional inverse of diagonal values
	for i in range(len(d)):
		M_inverse[i][i] = 1.0/d[i]
	def new_Afun(v):
		return np.dot(M_inverse,Afun(v))
	newB = np.dot(M_inverse,b)
	return cg(new_Afun, newB, tolerance)



#---- to plot graph comment out the below code and run, and wait for ~10 seconds--

# network = mn.makeNetwork('random2', 1000)
# b = getB(network)
# d = getDiag(network)
# x, residual_norms = cg(lambda v: applyA(network, v), -b, 10**-6)
# x1, residual_norms1 = pcg(lambda v: applyA(network, v), -b, d, 10**-6)
# ranges = np.arange(len(residual_norms))
# ranges1 = np.arange(len(residual_norms1))
# plt.plot(ranges,residual_norms, color='m', label="cg method")
# plt.plot(ranges1,residual_norms1, color='b', label="pcg method")
# plt.xlabel("n (step)")
# plt.legend()
# plt.ylabel("log( ||r|| / ||b|| )")
# plt.title("random2 network")
# plt.show()

#-------------------------------------------------------------------------