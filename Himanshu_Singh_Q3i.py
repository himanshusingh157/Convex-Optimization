#libraries
import cvxpy as cp
import numpy as np
import os
from oct2py import octave
#importing data
cwd=os.getcwd()
dirr=cwd+'/data_files/Problem3'
octave.addpath(dirr)
X,y=octave.svm_gendata(20,20,nout=2)
m,n=X.shape
#transformimg data
y=y.reshape(n)
Y=np.diag(y)
sigma=np.matmul(X.T,X)
P=-np.linalg.multi_dot([Y,sigma,Y])
c=np.ones(n)
#solving dual
lambdaa=cp.Variable(n,nonneg=True)
cost=c.T@lambdaa+cp.quad_form(lambdaa,-Y@sigma@Y)/2
constraints=[lambdaa>=0,y@lambdaa==0]

prob=cp.Problem(cp.Maximize(cost),constraints)
prob.solve()
#prininting dual optimal
print(prob.value)
