import cvxpy as cp
import numpy as np

n=200
d=10
#Generating Data
A=np.random.rand(n,d)
b=np.random.rand(n)
c=np.ones(n)

#Solving for the Optimal

#The L1 norm Problem with Linear program
x=cp.Variable((d))
t=cp.Variable((n))
cost=cp.Minimize(c.T@t)
constr=[A@x-b<=t,b-A@x<=t]
prob=cp.Problem(cost,constr)
prob.solve()
x11=x.value
print('###################L1 norm problem######################')
print('---------------With Linear Program------------------')
print(f'The optimal value is {prob.value}')
print(f"Solution for x is {x11}")
print('----------------------------------------------------')

#L1 norm Problem without Linear program
x=cp.Variable(d)
cost=cp.norm(A@x-b,1)
prob=cp.Problem(cp.Minimize(cost))
prob.solve()
x12=x.value
print('\n---------------Without Linear Program------------------')
print(f'The optimal value is {prob.value}')
print(f'Solution x is {x12}')
print('----------------------------------------------------\n')
error=np.linalg.norm(x11-x12)
print(f'Error between both Solution is {error}')


#L_inf norm Problem with Linear Program
x=cp.Variable(d)
p=cp.Variable(1)
cost=cp.Minimize(p)
constr=[A@x-b<=p*c,b-A@x<=p*c]
prob=cp.Problem(cost,constr)
prob.solve()
x21=x.value
print('\n#####################Infinite norm problem##################')
print('---------------With Linear Program------------------')
print(f'The optimal value is {prob.value}')
print(f'Solution x is {x21}')
print('----------------------------------------------------')

#L_inf norm Problem without Linear Program
x=cp.Variable(d)
prob=cp.Problem(cp.Minimize(cp.norm(A@x-b,'inf')))
prob.solve()
x22=x.value
print('\n---------------Without Linear Program------------------')
print(f'The optimal value is {prob.value}')
print(f'Solution x is {x22}')
error=np.linalg.norm(x21-x22)
print(f'Error between both Solution is {error}')
