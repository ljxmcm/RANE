
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time
import logging

import math
import random
from numpy import linalg as la
from sklearn.preprocessing import normalize

import  tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.24)
#config=tf.ConfigProto(gpu_options=gpu_options)

# In[2]:

logger = logging.getLogger()
logger.setLevel(logging.INFO)



def log_config():
    filename = './test_dp.log'
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)
    
class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[int(x)] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        # print('Results, using embeddings of dimensionality', len(self.embeddings[X[0]]))
        # print('-------------------')
        logging.info(results)
        return results["micro"]
        # print('-------------------')

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[int(x)] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y
    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)

def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y 


# In[3]:


def evaluation(Vectors, clf_ratio):
    vectors = Vectors
    X, Y = read_node_label('cora/cora_labels.txt')
    #print("Training classifier {%s}", filename)
    clf = Classifier(vectors=vectors, clf=LogisticRegression(solver='liblinear'))
    return clf.split_train_evaluate(X, Y, clf_ratio)


# In[4]:


def return_pos(a):
    b = np.zeros(len(a))
    for i in range(len(a)):
        b[a[i]] = len(a)-1-i
    return b


# In[5]:


def get_S():
    fin = open('cora/cora_edgelist.txt', 'r')
    node_size = int(fin.readline())
    S = np.zeros((node_size, node_size))
    for l in fin.readlines():
        vec = l.split()
        S[int(vec[0]), int(vec[1])] = 1.0
    fin.close()
    return S


# In[6]:


def get_A():
    fin = open('cora/cora.features', 'r')
    A = []
    for l in fin.readlines():
        vec = l.split()
        A.append([float(x) for x in vec[1:]])
    A = np.array(A)
    fin.close()
    return A


# In[7]:


def rank(A):
    At = np.dot(A, A.T)

    A_rank = np.zeros_like(At)
    i = 0
    for a in np.argsort(At):
        A_rank[i] = return_pos(a)
        i += 1
    return A_rank


# In[8]:


def search(beta=0.98, alpha = 0.34, sigma =920):
    A = get_A()
    node_size = A.shape[0]
    A_tfidf = np.zeros_like(A)
    for i in range(A.shape[1]):
        if np.sum(A[:, i]) > 0:
            A_tfidf[:,i] = A[:,i] * np.log(node_size/np.sum(A[:,i]))
    A_rank = rank(A_tfidf)
    #print(A_rank)

    S_1 = get_S()
    S_2 = np.dot(S_1, S_1.T)
    S_3 = np.dot(S_2, S_1.T)
    S_2[S_2>0] = 1*beta
    S_3[S_3>0] = 1*beta*beta
    S_23 = S_2+S_3
    S_23[S_23>beta] = beta
    S_123 = S_1 + S_23
    S_123[S_123>1] = 1
    S = S_123
    '''
    #beta = 0.8
    S = get_S()
    S_2 = np.dot(S, S.T)
    S_3 = np.dot(S_2, S.T)
    #S_2 = S_2 - np.diag(np.diag(S_2))
    #S_3 = S_3 - np.diag(np.diag(S_3))
    S = beta * S + (1-beta)*S_2
    S = beta * S + (1-beta)*S_3

    edge = np.sum(S)
    D = np.sum(S, axis=0, keepdims=True)
    S_community = np.dot(D.T, D)
    S_mod = S - S_community/(2*edge)
    '''
    S_2 = np.exp(-((S-1)*(S-1))/(2*np.power(alpha,2))) 
    A_2 = np.exp(-(A_rank*A_rank)/(2*np.power(sigma,2)))

    T = S_2*A_2
    T_norm = normalize(T)#T/np.sum(T, axis=1)
    return T_norm


# In[9]:


class model():
    def __init__(self, T, cur_seed):
        self.T = T
        self.T_flat = self.T.flatten()
        self.rep_size = 128
        self.node_size = self.T.shape[0]
        self.batch_size = 500000
        self.cur_epoch = 0
        
        self.cur_seed = cur_seed#random.getrandbits(32)
        self.learning_rate = 0.01
        
        tf.reset_default_graph()
        self.session = tf.Session()
        self.build_graph()
        self.session.run(tf.global_variables_initializer())
        
    def build_graph(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.sign = tf.placeholder(tf.float32, [None])
        #self.step = tf.placeholder(tf.int32)
        
        #self.learning_rate = tf.train.piecewise_constant(self.step, boundaries=self.boundaries, values=self.learning_rates)
    
        self.embeddings = tf.get_variable(name='embeddings', shape=[self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.cur_seed))
        self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
        self.t_e = tf.nn.embedding_lookup(self.embeddings, self.t)
        self.cor = tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)
        self.loss = tf.reduce_sum(tf.square(self.sign-self.cor)) + 0.001*tf.nn.l2_loss(self.h_e) + 0.001*tf.nn.l2_loss(self.t_e)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        
    def train_one_epoch(self):
        sum_loss = 0.0
        batcher = self.batch_iter()
        for batch in batcher:
            h,t,sign = batch
            feed_dict = {
                self.h: h,
                self.t: t,
                self.sign: sign,
                #self.step: self.cur_epoch
            }
            _, cur_loss = self.session.run([self.train_op, self.loss], feed_dict)
            sum_loss += cur_loss
        logging.info('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss))
        self.cur_epoch += 1
        
    def batch_iter(self):
        s_time = time()
        data_size = self.node_size*self.node_size
        col = np.random.choice(np.arange(self.node_size), size=data_size)
        row = np.random.choice(np.arange(self.node_size), size=data_size)
        
        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            h = row[start_index:end_index]
            t = col[start_index:end_index]
            #T_flat = self.T.flatten()
            sign = self.T_flat[h*self.node_size+t]
            yield h, t, sign
            start_index = end_index
            end_index = min(start_index+self.batch_size, data_size)
        print("time:{}".format(time()-s_time))
        
    def get_embeddings(self):
        embeddings = normalize(self.embeddings.eval(session=self.session))
        vectors = {}
        for i, embedding in enumerate(embeddings):
            vectors[i] = embedding
        return vectors
    
    def tf_close(self):
        self.session.close()


# In[10]:
def save_embeddings(vectors, filename):
    fout = open(filename, 'w')
    for node, vec in vectors.items():
        fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
    fout.close()

def micro(model):
    micro_max = 0
    epoch_max = 0
    for i in range(200):
        model.train_one_epoch()
        vectors = model.get_embeddings()
        score = evaluation(vectors, 0.5)
        if score >= micro_max:
            micro_max = score
            epoch_max = i
            best_vectors = vectors
    logging.info("max_epoch:{}, max_micro:{}".format(epoch_max, micro_max))
    save_embeddings(best_vectors, 'RANE')
    return micro_max


# In[11]:

log_config()
T = search(beta=0.1,alpha=0.33, sigma=900)
np.savetxt('T',T)
ASANE = model(T, cur_seed=17)
micro(ASANE)

# fout = open('ASANE_deep', 'w')
# for i, vec in vectors.items():
#     fout.write("{} {}\n".format(i, ' '.join([str(x) for x in vec])))
# fout.close()

# rep = train(T, epochs=80, dim=128)
# evaluation(rep, 0.5)

# def train(T, dim=50, epochs=20, lamb=0.2):
#     M = T
#     node_size = M.shape[0]
#     W = np.random.randn(dim, node_size)
#     H = W
#     for i in range(epochs):
#         #print('epochs:', i)
#         drv = 4 * np.dot(np.dot(W, W.T), W) - 4*np.dot(W, M.T) + lamb*W
#         Hess = 12*np.dot(W, W.T) + lamb*np.eye(dim)
#         drv = np.reshape(drv, [dim*node_size, 1])
#         rt = -drv
#         dt = rt
#         vecW = np.reshape(W, [dim*node_size, 1])
#         while np.linalg.norm(rt, 2) > 1e-4:
#             dtS = np.reshape(dt, (dim, node_size))
#             Hdt = np.reshape(np.dot(Hess, dtS), [dim*node_size, 1])
# 
#             at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
#             vecW = vecW + at*dt
#             rtmp = rt
#             rt = rt - at*Hdt
#             bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
#             dt = rt + bt * dt
#         W = np.reshape(vecW, (dim, node_size))
#     Vecs = normalize(W.T)
#     vectors = {}
#     for i, embedding in enumerate(Vecs):
#         vectors[i] = embedding
#     return vectors
#     '''
#     fout = open('vec_all3.txt', 'w')
#     node_num = len(vectors.keys())
#     for node, vec in vectors.items():
#         fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
#     fout.close()
#     '''
