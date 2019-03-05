import numpy as np
import networkx as nx

class R_Model():
    def __init__(self, edgelist, features):
        self.edgelist = edgelist
        self.A = self.load_topology(self.edgelist)
        self.X = self.load_attribute(features)
        self.node_size = self.A.shape[0]
        
    def load_topology(self, filename):
        fin = open(filename, 'r')
        node_size = int(fin.readline())
        Adj = np.zeros((node_size, node_size))
        for l in fin.readlines():
            vec = l.split()
            Adj[int(vec[0]), int(vec[1])] = 1.0
        fin.close()
        return Adj

    def load_attribute(self, filename):
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

    def return_pos(self, a):
        b = np.zeros(len(a))
        for i in range(len(a)):
            b[a[i]] = len(a)-1-i
        return b

    def rank_X(self):
        Sim = np.dot(self.X, self.X.T)
        R_x = np.zeros_like(Sim)
        i = 0
        for a in np.argsort(Sim):
            R_x[i] = self.return_pos(a)
            i += 1
        return R_x

    def rank_A(self, alpha = 0.1):
        unreach = 0
        G = nx.read_edgelist(self.edgelist)
        k = nx.all_pairs_shortest_path_length(G)
        dk = dict(k)
        SP = unreach*np.ones((self.node_size, self.node_size))
        for i in range(SP.shape[0]):
            dic = dk[str(i)]
            for j in range(SP.shape[1]):
                if str(j) in dic:
                    if i != j:
                        SP[i,j] = np.power(alpha, dic[str(j)]-1)
                    else:
                        SP[i,j] = np.power(alpha, 1)
        return SP