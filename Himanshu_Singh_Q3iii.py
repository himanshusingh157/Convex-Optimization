import cvxpy as cp
import numpy as np
import os
from oct2py import octave
import math
cwd=os.getcwd()
dirr=cwd+'/data_files/Problem3'
octave.addpath(dirr)
X,y=octave.svm_gendata(20,20,nout=2)
sigma_values=[1e-2,1e-1,0.5,10,100]
y=y.reshape(y.shape[1])
Y=np.diag(y)
c=np.ones(y.shape[0])
def fill_sigma(sig,X_mat,m):
    mat=np.zeros((m,m))
    for i in range(m):
        for j in range(i,m):
            p=np.linalg.norm(X[:,i]-X[:,j])
            p=(-p/sig**2)/2
            mat[i,j]=math.exp(p)
            mat[j,i]=mat[i,j]
    return mat

error=np.zeros(len(sigma_values))
for i in range(len(sigma_values)):
    sigma=fill_sigma(sigma_values[i],X,y.shape[0])
    lambdaa=cp.Variable(n,nonneg=True)
    cost=c.T@lambdaa+cp.quad_form(lambdaa,-Y@sigma@Y)/2
    constraints=[lambdaa>=0,y@lambdaa==0]
    prob=cp.Problem(cp.Maximize(cost),constraints)
    prob.solve()
    error[i]=prob.value
print(error)

