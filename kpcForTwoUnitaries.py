#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:57:27 2024

@author: dubhaltachmcsweeney
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import tensordot as tensor
from numpy import trace
from scipy.linalg import sqrtm,eigvals,pinv, norm
from numpy.linalg import matrix_power, inv, det, qr
from numpy.random import uniform
import pennylane as qml
import array_to_latex as a2l #a useful library

trace_permutations = [[3,1],[3,0],[2,0],[2,1]]

"""
- - - - - - - - - - - - - - - - -
General Linear Algebra Functions
- - - - - - - - - - - - - - - - -
"""
def idi(N):
    "A function to make identity matrices"
    return np.diag(np.ones(N))

def trace_x(rho,sys,dim):
    """
    
    Parameters
    ----------
    rho:    matrix containing the quantum state that is to be partially traced over
    sys:    list of spaces with respect to which the partial trace is taken
    dim:    list of dimensions of the spaces rho is defined on 
    
    Requires
    ----------
    numpy as np
    
    Returns
    ----------
    Partial trace of rho with respect to the systems given in sys. 
    
    """
    
    #Dimensions of the traced out and the remaining systems
    D = np.prod(dim)
    Dtrace = np.prod(np.array(dim)[sys])
    Dremain = int(D/Dtrace)
    
    #shape required for simple tracing
    shfinal = [Dremain, Dtrace, Dremain, Dtrace]
    
    #parameters required to decompose rho into its subsystems
    le = len(dim)  
    arshape = np.append(dim,dim)
    
    #permutation to permute all spaces that need to be traced out to the right, 
    #so that they can be traced out in one go.
    perm = np.arange(le)
    perm = np.append(np.delete(perm,sys),np.array(sys))
    perm = np.append(perm,perm+le)
    
    #reshape rho, permute all spaces that need to be traced out to the right, 
    #reshape into the form [Dremain, Dtrace, Dremain, Dtrace] and trace out over
    #the Dtrace parts.
    return np.trace(rho.reshape(arshape).transpose(perm).reshape(shfinal), axis1=1, axis2=3)
    
def sys_permute(rho,perm,dim):
    """
    
    Parameters
    ----------
    rho:    matrix containing the quantum state that is to be partially traced over
    perm:   list defining the permutation of spaces
    dim:    list of dimensions of the spaces rho is defined on 
    
    Requires
    ----------
    numpy as np
    
    Returns
    ----------
    Permutation of rho according to the permutation given in perm
	
	"""
	
    le = len(dim)	
    sh = rho.shape
	#array required for the reshaping of rho so that all indeces are stored 
	#in different axes
    arshape = np.append(dim,dim)
	
	#Array for correct permutation 
    P = np.append(np.array(perm),np.array(perm)+le)
    return rho.reshape(arshape).transpose(P).reshape(sh)

def tn(*args, sparse=False):
    """
    Parameters
    ----------
    args: elements which are to be tensored
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    tensor product of the elements varargin
    
    """
    if len(args) == 0:
        print('tn_product requires at least one argument. Returned an empty array.')
        return np.array([])
    else:
        result = args[0]
        for Mat in args[1:]:
            if sparse:
                result = sparse.kron(result, Mat)
            else:
                result = np.kron(result,Mat)
        return result

def dagger(M):
    """
    Parameters
    ----------
    M: two dimensional matrix
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    transpose conjugate of input matrix M
    """

    return np.transpose(np.conjugate(M))


"""
- - - - - - - - - - - - -
Kraus, Petz and Choi
- - - - - - - - - - - - -
"""

"""Kraus calculation and use"""

def find_kraus(U,env):
    """
    only for 8x8 (2,2,2) but could easily me modified
    Parameters
    ----------
    U: two dimensional unitary matrix
    env: vector of enviroment state
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    Kraus operators in matrix form
    """
    K = []
    env_basis = [zero,one]#this is all in terms of qubits
    for e in env_basis:
        A = sys_permute(tn(tensor(e,env,axes=0),idi(4)),(1,0,2),(2,2,2))
        K.append(trace_x(A@U,1,(2,2,2))) #again assuming 8x8 
    return K

def kraus_map(K,state):
    """
    Parameters
    ----------
    K: Set of Kraus operators
    state: Input state
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    Mapping
    """
    out = []
    for k in K:
        out.append(k@state@dagger(k)) #straight forward enough

    return sum(out)

def hermitian_kraus_map(K,state):
    """
    Parameters
    ----------
    K: Set of Kraus operators
    state: Input state
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    Hermitian Mapping
    """
    out = []
    for k in K:
        out.append(dagger(k)@state@k) #straight forward enough
      
    return sum(out)


"""Petz for Kraus operators"""

def petz(X,K,S):
    """
    Parameters
    ----------
    K: Set of Kraus operators
    X: Input state
    S: mapping state of Petz
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    Petz Mapping
    """
    
    return sqrtm(S) @ hermitian_kraus_map( K , ((pinv(sqrtm(kraus_map(K,S))))@X@(pinv(sqrtm(kraus_map(K,S)))))) @ sqrtm(S)

"Choi and petz choi"

def choi(K):
    """
    Parameters
    ----------
    K: Unitary operator of map
    -------
    numpy as np
    
    Returns
    -------
    Choi matrix
    
    """
    
    C = 0
    
    basis = np.diag(np.ones(2))
    for i in basis:
        for j in basis:
            for k in basis:
                for l in basis:
                    state = tn(tensor(i,j,axes=0),tensor(k,l,axes=0))
                    C += tn(kraus_map(K,state),state)
    return C
    
def petz_choi(K,S):
    """
    Parameters
    ----------
    K: Unitary operator of map
    S: Petz map state
    -------
    numpy as np
    
    Returns
    -------
    Choi matrix for petz map
    
    """
    C = 0
    
    basis = np.diag(np.ones(2))
    for i in basis:
        for k in basis:
            for j in basis:
                for l in basis:
                    state = tn(tensor(k,i,axes=0),tensor(l,j,axes=0))
                    C += tn(petz(state,K,S),state)
    return C

"""
- - - - - - - - - - - - - - - - - - - - - - - -
Sample of random matrices with the haar measure
- - - - - - - - - - - - - - - - - - - - - - - -
"""

def qr_haar(N): #this works great
    """
    Taken from pennylane tutorial
    
    Parameters
    ----------
    N: dimension of desired square matrix
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    A Haar-random matrix generated using the QR decomposition
    """
    # Step 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + (1j * B)

    # Step 2
    Q, R = qr(Z)

    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4
    return np.dot(Q, Lambda)

dev = qml.device('default.mixed', wires=2) #this doesn't really work
@qml.qnode(dev)

def qr_haar_random_unitary(): #this does something strange that I don't and doesn't work
    qml.QubitUnitary(qr_haar(2), wires=(0)) #this makes sense you need at least 2 wires for a 4x4
    return qml.state()

def haar_sample(N,n):
    """
    Taken from pennylane tutorial
    
    Parameters
    ----------
    N: Largest
    
    Requires
    -------
    numpy as np
    
    Returns
    -------
    Returns random sample of 4x4
    """
    qr_haar_samples = [qr_haar_random_unitary() for _ in range(N)] #these are my 2024 random samples
    
    qr_haar_samples = [qr_haar(n) for _ in range(N)]
    
    return qr_haar_samples

def special_samples(N):
    
    qr_haar_samples = [qr_haar(4) for _ in range(N)]
    
    qr_haar_samples = [i/det(i) for i in qr_haar_samples]
    
    return qr_haar_samples
    
    
"""
- - - - - - - - - - - - - - - - - - - - - - - -
Super function that:
takes two unitary matrices, 
makes our causally ordered map,
finds kraus operators (i.e. "discards" enviroment),
makes Petz Choi, and
traces out all possible combination to search for causal order
- - - - - - - - - - - - - - - - - - - - - - - -
"""

def super_function(u1,u2,env,S):
    """
    
    Parameters
    ----------
    u1:     first unitary operator in causal sequence
    u2:     second operator in causal sequence
    env:    enviroment used to generate Kraus operators / enviroment discarded
    S:      choi state, 4x4 density operator
    
    Requires
    ----------
    Everything above
    
    Returns
    ----------
    List of norm(C_abc - C_bc), List of Petz norms, Choi, Petz Choi, U, K
	
	"""
    
    #   (gate flipped wrt input x wire)       @  (wire x gate)
    
    U = tn(sys_permute(u2,(1,0),(2,2)),idi(2))@tn(idi(2),u1)
    
    K = find_kraus(U,env) #find kraus
    
    C = choi(K) #make choi
    
    C_p = petz_choi(K,S) #make petz choi
    
    re_add_space_dict = {2:(0,1,2),1:(0,2,1),0:(2,0,1)} #it matters where you tensor back in the identity
    
    norms = []
    for p in trace_permutations: #using set of possible out, in trace orders at top of code
        n = trace_x(C,p[0],(2,2,2,2))
        nn = trace_x(n,p[1],(2,2,2))
        norms.append((norm(n - 0.5*sys_permute(tn(nn,idi(2)),re_add_space_dict[p[1]],(2,2,2)))))
    
    petz_norms = []
    for p in trace_permutations:
        n = trace_x(C_p,p[0],(2,2,2,2))
        nn = trace_x(n,p[1],(2,2,2))
        petz_norms.append(norm(n - (0.5)*sys_permute(tn(nn,idi(2)),re_add_space_dict[p[1]],(2,2,2))))
                         
    return norms, petz_norms, C, C_p, U, K #returning everythin useful

"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
And now using the super_function on some special and random unitaries
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""

"""Standard qubit states"""

zero = idi(2)[0]
one = idi(2)[1]

plus = (1/np.sqrt(2))*(zero + one)
minus = (1/np.sqrt(2))*(zero - one)

"""Special cases of unitary"""

CNOT = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]])

SWAP = np.array([[1,0,0,0],
                 [0,0,1,0],
                 [0,1,0,0],
                 [0,0,0,1]])

"""Haar random unitaries"""

random_U = qr_haar(4)
random_Ub = qr_haar(4)

r_u = qr_haar(2)
C_U = np.array([[1,0,0,0],
                [0,1,0,0,],
                [0,0,r_u[0][0],r_u[0][1]],
                [0,0,r_u[1][0],r_u[1][1]]])

"""Our petz state"""

S = tn(tensor(one,one,axes=0),tensor(zero,zero,axes=0))

"""Calling the super function"""

r = super_function(CNOT,CNOT,plus,S)

"""
- - - - - - - - - - - - - - - - - - - - -
plotting to might visualise results
- - - - - - - - - - - - - - - - - - - - -
I hate matplotlib, all plotting below is messy.
"""

"""
For testing single channels
"""

spacenames = [str(i) for i in trace_permutations]

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),dpi=200)

ax1.barh(spacenames, r[0], color='darkgreen')
ax1.set_title("2-step CNOT Quantum Channel")
ax1.set_xlabel('')
ax1.title.set_size(20)

ax2.barh(spacenames,r[1], color='blue')
ax2.set_yticks([])
ax2.set_title("Petz Map")
ax2.set_ylabel('')
ax2.title.set_size(20)

plt.savefig('../../../Documents/CapstoneReport/CNOTvsPetz.pdf')
plt.show()


"""
For testing average number of channels
"""

num_samples = 1000
random_sample_for_CU = haar_sample(num_samples,2)
random_CU = [np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,i[0][0],i[0][1]],
                      [0,0,i[1][0],i[1][1]]]) for i in random_sample_for_CU]

random_sample = haar_sample(num_samples,4)

N = []
N_p = []

for i in range(int(num_samples/2)):
    s = super_function(random_sample[i],random_sample[i+(int(num_samples/2))],one,S)
    a = s[0]
    b = s[1]
    N.append(np.array(a))
    N_p.append(np.array(b))
    
N_p = np.array(sum(N_p))*(1/num_samples)
N = np.array(sum(N))*(1/num_samples)
  
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),dpi=200)

ax1.barh(spacenames, N, color='darkgreen', label='Chart 1')
ax1.set_title("Average 1000 samples of 2-step Random U")
ax1.set_xlabel('')
ax1.title.set_size(20)

ax2.barh(spacenames,N_p, color='blue', label='Chart 2')
ax2.set_yticks([])
ax2.set_title("Average of Petz Maps ")
ax2.set_ylabel('')
ax2.title.set_size(20)


plt.tight_layout()

plt.savefig('../../../Documents/CapstoneReport/RandomUvsPetz.pdf')
plt.show()


