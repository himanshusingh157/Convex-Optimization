#importing libraries
import cvxpy as cp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import math

#For this particular example we take 50*50 matrix
m=50
n=50

#Generatinf a low rank matrix
L_gen=np.random.rand(m,n)
U,Sv,V=np.linalg.svd(L_gen)
Sv[5:]=Sv[5:]*0
L_gen=(U@np.diag(Sv))@V
#genearting a sparse matrix
S_gen=sparse.random(m,n,density=0.1)
S_gen=S_gen.toarray()
#M=L+S
M=L_gen+S_gen

lambdaa=list(range(1,10))
error=np.zeros((2,len(lambdaa))) #for storing the error

for i in range(len(lambdaa)):
    L=cp.Variable((m,n))
    S=cp.Variable((m,n))
    cost=cp.norm(L,"nuc")+lambdaa[i]*cp.norm(S,1)
    constr=[L+S==M]
    prob=cp.Problem(cp.Minimize(cost),constr)
    prob.solve()
    error[0,i]=np.linalg.norm(L.value-L_gen,'fro')
    error[1,i]=np.linalg.norm(S.value-S_gen,'fro')

#plotting error
plt.plot(lambdaa,error[0])
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.title('Error between Generated low rank matrix and predicted low rank matrix')
plt.show()

plt.plot(lambdaa,error[1])
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.title('Error between Generated sparse matrix and predicted sparse matrix')
plt.show()





#Now taking value of lambda = sqrt(50)
lambdaa=math.sqrt(50)
L=cp.Variable((m,n))
S=cp.Variable((m,n))
cost=cp.norm(L,"nuc")+lambdaa*cp.norm(S,1)
constr=[L+S==M]
prob=cp.Problem(cp.Minimize(cost),constr)
prob.solve()
error1=np.linalg.norm(L.value-L_gen,'fro')
error2=np.linalg.norm(S.value-S_gen,'fro')

print('When lambda was take to be sqrt(n) the error between original matrix and predicted matrix were:')
print(f'Error between original Low rank matrix and predicted low rank matrix is {error1}')
print(f'Error between original sparse matrix and predicted sparse matrix is {error2}')
