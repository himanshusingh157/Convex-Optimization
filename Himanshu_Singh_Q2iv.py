#importing libraries
import cvxpy as cp
import numpy as np
import os
from oct2py import octave
import matplotlib.pyplot as plt
import math

#setting up environment
sigma=np.array(list(range(1,21)))
sigma=sigma/10
cwd=os.getcwd()
dirr=cwd+'/data_files/Problem2'
octave.addpath(dirr)
lambdaa=5
m=200
n=50
error=np.zeros((3,len(sigma)))

#Iterating over values of sigma
for i in range(len(sigma)):
    A,b=octave.gendata_lasso(200,50,sigma[i],2,nout=2)
    x=cp.Variable((n,1))
    y=cp.Variable((n,1))
    z=cp.Variable((n,1))
    cost1=cp.sum_squares(A@x-b)
    cost2=cp.sum_squares(A@y-b)+lambdaa*cp.sum_squares(y)
    cost3=cp.sum_squares(A@z-b)+lambdaa*cp.norm(z,1)
    prob1=cp.Problem(cp.Minimize(cost1))
    prob2=cp.Problem(cp.Minimize(cost2))
    prob3=cp.Problem(cp.Minimize(cost3))
    prob1.solve()
    prob2.solve()
    prob3.solve()
    error[0,i]=np.linalg.norm((A@x.value)-b)
    error[1,i]=np.linalg.norm((A@y.value)-b)
    error[2,i]=np.linalg.norm((A@z.value)-b)

error=error/math.sqrt(m)
#plotting errors

p=np.arange(len(sigma))
width=0.25
fig,ax=plt.subplots()
rects1=ax.bar(p-width,error[0,:],width,label='Least Square fitting error')
rects1=ax.bar(p,error[1,:],width,label='L2 Regularization error')
rects2=ax.bar(p+width,error[2,:],width,label='L1 Regularization error')
ax.set_ylabel('Error')
ax.set_xlabel('Sigma')
ax.set_title('Root Mean Square vs Sigma')
ax.set_xticks(p)
ax.set_xticklabels(sigma)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()

