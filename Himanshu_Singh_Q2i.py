#importing Libraries
import cvxpy as cp
import numpy as np
import os
from oct2py import octave
import matplotlib.pyplot as plt
import sys
#getting data
cwd=os.getcwd()
dirr=cwd+'/data_files/Problem2'
octave.addpath(dirr)
A,b=octave.gendata_lasso(nout=2)
m,n=A.shape
lambdaa=5


#Least square fitting
x=cp.Variable((n,1))
cost=cp.sum_squares(A @ x - b)
prob=cp.Problem(cp.Minimize(cost))
#prob.solve(solver=cp.ECOS,verbose=True)
#Saving the putput in text file and then
#extracting the pro=imary cost to plot
out=sys.stdout
sys.stdout=open('output.txt','w')
a=prob.solve(solver=cp.ECOS,verbose=True)
sys.stdout.close()
sys.stdout=out
f=open('output.txt','r')
linenum=0
pcost=[]
for line in f:
    linenum+=1
    if linenum < 5:
        continue
    if line == '\n':
        break
    cols=line.split(' ')
    if linenum >=15:
        pcost.append(float(cols[2]))
    else:
        pcost.append(float(cols[3]))

#output
print('################## Least Square fitting ##############') 
print('Iteration \t primary cost')
for i in range(1,len(pcost)):
    print(f'  {i}\t        {pcost[i]}')
q=list(range(1,len(pcost)))
#plotting
plt.plot(q,pcost[1:])
plt.xlabel('Iteration count')
plt.ylabel('Objective Value')
plt.title('Linear leastsquare cost vs iteration')
plt.show()
r=np.log(np.abs(pcost[1:]))
plt.plot(q,r)
plt.ylabel('Log of absolute value of Objective function')
plt.xlabel('Iteration count')
plt.title('Linear leastsquare cost vs iteration')
plt.show()



#L2 regularization
x=cp.Variable((n,1))
cost=cp.sum_squares(A @ x - b) +lambdaa*cp.sum_squares(x)
prob=cp.Problem(cp.Minimize(cost))
#Saving the output in text file and then
#extracting the pro=imary cost to plot
out=sys.stdout
sys.stdout=open('output.txt','w')
a=prob.solve(solver=cp.ECOS,verbose=True)
sys.stdout.close()
sys.stdout=out

f=open('output.txt','r')
linenum=0
pcost=[]
for line in f:
    linenum+=1
    if linenum < 5:
        continue
    if line == '\n':
        break
    cols=line.split(' ')
    if linenum >=15:
        pcost.append(float(cols[2]))
    else:
        pcost.append(float(cols[3]))
#output
print('################## L2 Regularization ##############') 
print('Iteration \t primary cost')
for i in range(1,len(pcost)):
    print(f'  {i}\t        {pcost[i]}')
q=list(range(1,len(pcost)))
#plotting
plt.plot(q,pcost[1:])
plt.xlabel('Iteration count')
plt.ylabel('Objective Value')
plt.title('L2 Regularization cost vs iteration')
plt.show()
r=np.log(np.abs(pcost[1:]))
plt.plot(q,r)
plt.ylabel('Log of absolute value of Objective function')
plt.xlabel('Iteration count')
plt.title('L2 Regularlization cost cost vs iteration')
plt.show()





#L1 regularization
x=cp.Variable((n,1))
cost=cp.sum_squares(A @ x - b) +lambdaa*cp.norm(x,1)
prob=cp.Problem(cp.Minimize(cost))
#Saving the putput in text file and then
#extracting the pro=imary cost to plot
out=sys.stdout
sys.stdout=open('output.txt','w')
a=prob.solve(solver=cp.ECOS,verbose=True)
sys.stdout.close()
sys.stdout=out

f=open('output.txt','r')
linenum=0
pcost=[]
for line in f:
    linenum+=1
    if linenum < 5:
        continue
    if line == '\n':
        break
    cols=line.split(' ')
    if linenum >=15:
        pcost.append(float(cols[2]))
    else:
        pcost.append(float(cols[3]))
#output
print('################## L1 Regularization ##############') 
print('Iteration \t primary cost')
for i in range(1,len(pcost)):
    print(f'  {i}\t        {pcost[i]}')
q=list(range(1,len(pcost)))
#plotting
plt.plot(q,pcost[1:])
plt.xlabel('Iteration count')
plt.ylabel('Objective Value')
plt.title('L2 Regularization cost vs iteration')
plt.show()
r=np.log(np.abs(pcost[1:]))
plt.plot(q,r)
plt.ylabel('Log of absolute value of Objective function')
plt.xlabel('Iteration count')
plt.title('L1 Regularlization cost cost vs iteration')
plt.show()

