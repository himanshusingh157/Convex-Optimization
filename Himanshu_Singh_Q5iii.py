#importing Libraries
import cvxpy as cp
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
#Generating a bipartite graph
m=8
n=12
G = nx.Graph()
G.add_nodes_from(list(range(m+n)))
V1_true=list(range(m+n))
np.random.shuffle(V1_true)
V2_true=V1_true[m:]
V1_true=V1_true[:m]
V1_true.sort()
V2_true.sort()
W=np.zeros((m+n,m+n))
edges=list()
for i in range(m):
    for j in range(n):
        W[V1_true[i],V2_true[j]]=1
        W[V2_true[j],V1_true[i]]=1
        edges.append((V1_true[i],V2_true[j]))
#creating Laplacian Matrix
D=np.sum(W,axis=0)
D=np.diag(D)
L=D-W

c=np.ones(m+n)
#Solving the optimization problem
X=cp.Variable((m+n,m+n),PSD=True)
cost=(0.25)*cp.trace(L@X)
constr=[X>>0, cp.diag(X)==c]
prob=cp.Problem(cp.Maximize(cost),constr)
prob.solve(solver=cp.CVXOPT)

#Generating labels
M=np.linalg.cholesky(X.value)
u=np.random.uniform(-1,1,m+n)
u=u/np.linalg.norm(u)

labels=M@u
for i in range(m+n):
    if labels[i]>=0:
        labels[i]=1
    else:
        labels[i]=-1

#Calculating weight of predicted graph
weight=0            
for i in range(m+n):
    for j in range(i+1,m+n):
        if labels[i]!=labels[j]:
            weight=weight+W[i,j]

print(f'weight of graoh generated is {weight}')

V1_pred=[]
V2_pred=[]
for i in range(m+n):
    if labels[i]==1:
        V1_pred.append(i)
    else:
        V2_pred.append(i)

print(f'The original set V1 of graph is {V1_true}')
print(f'The original set V2 of graph is {V2_true}')
print(f'The predicted set V1 of graph is {V1_pred}')
print(f'The predicted set V2 of graph is {V2_pred}')

#true graph
G.add_edges_from(edges)
X,Y=nx.bipartite.sets(G)
pos = dict()
pos.update( (n, (1, i)) for i, n in enumerate(X) ) 
pos.update( (n, (2, i)) for i, n in enumerate(Y) ) 
nx.draw(G, pos=pos,with_labels=True)
plt.show()

#predicted graph
G = nx.Graph()
G.add_nodes_from(list(range(m+n)))
edges=list()
for i in range(len(V1_pred)):
    for j in range(len(V2_pred)):
        if W[V1_pred[i],V2_pred[j]]!=0:
            edges.append((V1_pred[i],V2_pred[j]))
        
G.add_edges_from(edges)
nx.draw(G, pos=pos,with_labels=True)
plt.show()

