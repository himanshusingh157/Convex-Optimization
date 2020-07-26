#importing Libraries 
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#taking m random points in [-1,1]
m=10
a=-1+2*np.random.rand(m)
a=np.sort(a)
c=np.ones(m)
#Maximizing the entropy to find the probability distribution
p=cp.Variable(m)
cost=cp.sum(cp.entr(p))
constraints=[p>=0,c@p==1]
prob=cp.Problem(cp.Maximize(cost),constraints)
prob.solve()
p_pred=p.value
p_pred= np.round_(p_pred,decimals = 5)
#plotting
plt.plot(a,p_pred)
plt.xticks(np.arange(-1,1.01,0.25))
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Probability Distribution')
plt.show()

