#Libraries
import cvxpy as cp
import numpy as np
import os
from oct2py import octave
#importing graph
cwd=os.getcwd()
dirr=cwd+'/data_files/Problem5'
octave.addpath(dirr)
n=50
W=octave.rgg(n)
D=np.sum(W,axis=0)
D=np.diag(D)
L=D-W #laplacian matrix

c=np.ones(n)
#solcing optimization problem
X=cp.Variable((n,n),PSD=True)
cost=(0.25)*cp.trace(L@X)
constr=[X>>0, cp.diag(X)==c]
prob=cp.Problem(cp.Maximize(cost),constr)
prob.solve(solver=cp.CVXOPT)
print(prob.value)
rank=np.linalg.matrix_rank(X.value) #rank of predicte matrix
print(f'Rank={rank}')

M=np.linalg.cholesky(X.value) #cholesky decomposition
u=np.random.uniform(-1,1,n)
u=u/np.linalg.norm(u)

labels=M@u  #labels
for i in range(n):
    if labels[i]>=0:
        labels[i]=1
    else:
        labels[i]=-1

weight=0   #maxcut of the graph obtained    
for i in range(n):
    for j in range(i+1,n):
        if labels[i]!=labels[j]:
            weight=weight+W[i,j]

print(f'Maxcut of the problem = {weight}')

