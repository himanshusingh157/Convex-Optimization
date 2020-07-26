#importing all the required libraries
import cvxpy as cp
import numpy as np
import os
from oct2py import octave
import matplotlib.pyplot as plt

#Loading A,b
cwd=os.getcwd()
dirr=cwd+'/data_files/Problem2'
octave.addpath(dirr)
A,b=octave.gendata_lasso(nout=2)
m,n=A.shape

#lambdaa
lambdaa=list(range(1,21))
lambdaa=np.array(lambdaa)*5

#matrix for storing data
P=np.zeros((20,n))
Q=np.zeros((20,n))
R=np.zeros((20,2))
S=np.zeros((20,2))

#for different values of lambda
for i in range(len(lambdaa)):
    x=cp.Variable((n,1))
    y=cp.Variable((n,1))
    cost11 = cp.sum_squares(A @ x - b)
    cost12 = lambdaa[i]*cp.sum_squares(x)
    cost21 = cp.sum_squares(A @ y - b)
    cost22 = lambdaa[i]*cp.norm(y,1)
    prob1 = cp.Problem(cp.Minimize(cost11+cost12))
    prob2 = cp.Problem(cp.Minimize(cost21+cost22))
    prob1.solve()
    prob2.solve()
    P[i,:]=x.value.T
    Q[i,:]=y.value.T
    R[i,0]=cost11.value
    R[i,1]=cost12.value
    S[i,0]=cost21.value
    S[i,1]=cost22.value
#plotting
#plotting first 10 coordinates of the optimal 'x' for different values of lambda
for i in range(10):
    plt.plot(lambdaa,P[:,i],linewidth=1,label='Coordinate'+str(i+1))
plt.xlabel('lambda')
plt.ylabel('Optimal x')
plt.title('Minimizer of L2 reguralization vs lambda for first 10 coordinates')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
#plotting all the coordinates of the optimal 'x' for different values of lambda
for i in range(n):
    plt.plot(lambdaa,P[:,i],linewidth=1)
plt.xlabel('lambda')
plt.ylabel('Optimal x')
plt.title('Minimizer of L2 reguralization vs lambda')
plt.show()

#plotting first 10 coordinates of the optimal 'x' for different values of lambda
for i in range(10):
    plt.plot(lambdaa,Q[:,i],linewidth=1,label='Coordinate'+str(i+1))
plt.xlabel('lambda')
plt.ylabel('Optimal x')
plt.title('Minimizer of L1reguralization vs lambda  for first 10 coordinates')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
#plotting all the coordinates of the optimal 'x' for different values of lambda
for i in range(n):
    plt.plot(lambdaa,Q[:,i],linewidth=1)
plt.xlabel('lambda')
plt.ylabel('Optimal x')
plt.title('Minimizer of L1reguralization vs lambda')
plt.show()

#plotting objective function for different values of lambda
plt.plot(lambdaa,R[:,0],c='g',label="Least Square term term")
plt.plot(lambdaa,R[:,1],c='r',label='L2 Regularization term')
plt.plot(lambdaa,R[:,1]+R[:,0],c='b',label='Total cost of function')
plt.xlabel('lambda')
plt.ylabel('Cost')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
#plotting objective function for different values of lambda
plt.plot(lambdaa,S[:,0],c='g',label="Least Square term")
plt.plot(lambdaa,S[:,1],c='r',label='L1 Regularization term')
plt.plot(lambdaa,S[:,1]+S[:,0],c='b',label='Total cost of function')
plt.xlabel('lambda')
plt.ylabel('Cost')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


