import numpy as np
import networkx as nx

def load_topology(filename):
    fin = open(filename, 'r')
    node_size = int(fin.readline())
    Adj = np.zeros((node_size, node_size))
    for l in fin.readlines():
        vec = l.split()
        Adj[int(vec[0]), int(vec[1])] = 1.0
    fin.close()
    return Adj

def load_attribute(filename):
    fin = open(filename, 'r')
    X = []
    for l in fin.readlines():
        vec = l.split()
        X.append([float(x) for x in vec[1:]])
    fin.close()
    X =  np.array(X)
    node_size =X.shape[0]
    X_tfidf = np.zeros_like(X)
    for i in range(X.shape[1]):
        if np.sum(X[:, i]) > 0:
            X_tfidf[:,i] = X[:,i] * np.log(node_size/np.sum(X[:,i]))
    return X_tfidf

def return_pos(a):
    b = np.zeros(len(a))
    for i in range(len(a)):
        b[a[i]] = len(a)-1-i
    return b

def rank_X(X):
    Sim = np.dot(X, X.T)

    R_x = np.zeros_like(Sim)
    i = 0
    for a in np.argsort(Sim):
        R_x[i] = return_pos(a)
        i += 1
    return R_x

def rank_A(S_1):
    unreach = 0
    alpha = 0.1
    
    G = nx.read_edgelist('cora/cora_edgelist.txt')
    k = nx.all_pairs_shortest_path_length(G)
    dk = dict(k)
    SP = unreach*np.ones((2708,2708))
    for i in range(SP.shape[0]):
        dic = dk[str(i)]
        for j in range(SP.shape[1]):
            if str(j) in dic:
                if i != j:
                    SP[i,j] = np.power(alpha, dic[str(j)]-1)
                else:
                    SP[i,j] = np.power(alpha, 1)
    '''
    S_2 = np.dot(S_1, S_1.T)
    S_3 = np.dot(S_2, S_1.T)
    S_2[S_2>0] = alpha
    S_3[S_3>0] = alpha*alpha
    S_23 = S_2+S_3
    S_23[S_23>alpha] = alpha
    S_123 = S_1 + S_23
    S_123[S_123>1] = 1
    S = S_123
    '''
    return SP

def Gaussin(v, a, b, c):
    return a*np.exp(-((v-b)*(v-b))/(2*np.power(c,2)))