#libraries
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
#sampling
m=20
a=-1+(2*np.random.rand(m))
a=np.sort(a)
c=np.ones(m)

#Solving
p=cp.Variable(m)
cost=cp.sum(cp.entr(p))
constraints=[p>=0,c@p==1]
prob=cp.Problem(cp.Maximize(cost),constraints)
prob.solve()
#solution without any expectation constraint
print(prob.value)
p_pred1=p.value

print(f'Predicted values of Probability when no expectation constraint was there {p_pred1}')
print(f'And entropy is {p_pred1}')
#solution with expectation constraint
p=cp.Variable(m)
cost=cp.sum(cp.entr(p))
q=np.power(a,3)-(2*a)
r=[int(a[i]<0.5) for i in range(m)]
constraints=[p>=0,c@p==1,p@a>=-0.1,p@a<=0.1,np.square(a)@p>=0.5,np.square(a)@p<=0.5,q@p>=-0.3,q@p<=-0.2,r@p>=0.3,r@p<=0.4]
prob=cp.Problem(cp.Maximize(cost),constraints)
prob.solve()
print(prob.status)
p_pred2=p.value
print(p_pred2)
print(f'Predicted values of Probability on adding expectation constraint {p_pred2}')
print(f'And entropy is {p_pred2}')
plt.plot(a,p_pred2)
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Ramdom variable under all given constraints')
plt.xticks(np.arange(-1,1.01,0.25))
plt.show()

