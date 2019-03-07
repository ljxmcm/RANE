import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize

#GPU training
class T_Model():
    def __init__(self, T):
        self.T = T
        self.T_flat = self.T.flatten()
        self.rep_size = 128
        self.node_size = self.T.shape[0]
        self.batch_size = 500000
        self.cur_epoch = 0
        
        self.cur_seed = 0
        self.learning_rate = 0.01
        
        tf.reset_default_graph()
        self.session = tf.Session()
        self.build_graph()
        self.session.run(tf.global_variables_initializer())
        
    def build_graph(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.sign = tf.placeholder(tf.float32, [None])
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
            }
            _, cur_loss = self.session.run([self.train_op, self.loss], feed_dict)
            sum_loss += cur_loss
        #print('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss))
        self.cur_epoch += 1
        
    def batch_iter(self):
        data_size = self.node_size*self.node_size
        col = np.random.choice(np.arange(self.node_size), size=data_size)
        row = np.random.choice(np.arange(self.node_size), size=data_size)
        
        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            h = row[start_index:end_index]
            t = col[start_index:end_index]
            sign = self.T_flat[h*self.node_size+t]
            yield h, t, sign
            start_index = end_index
            end_index = min(start_index+self.batch_size, data_size)
        
    def get_embeddings(self):
        embeddings = normalize(self.embeddings.eval(session=self.session))
        vectors = {}
        for i, embedding in enumerate(embeddings):
            vectors[i] = embedding
        return vectors

#CPU training Matrix Factorization 
def train(T, dim=50, epochs=20, lamb=0.2):
    M = T
    node_size = M.shape[0]
    W = np.random.randn(dim, node_size)
    H = W
    for i in range(epochs):
        #print('epochs:', i)
        drv = 4 * np.dot(np.dot(W, W.T), W) - 4*np.dot(W, M.T) + lamb*W
        Hess = 12*np.dot(W, W.T) + lamb*np.eye(dim)
        drv = np.reshape(drv, [dim*node_size, 1])
        rt = -drv
        dt = rt
        vecW = np.reshape(W, [dim*node_size, 1])
        while np.linalg.norm(rt, 2) > 1e-4:
            dtS = np.reshape(dt, (dim, node_size))
            Hdt = np.reshape(np.dot(Hess, dtS), [dim*node_size, 1])

            at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
            vecW = vecW + at*dt
            rtmp = rt
            rt = rt - at*Hdt
            bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
            dt = rt + bt * dt
        W = np.reshape(vecW, (dim, node_size))
    Vecs = normalize(W.T)
    vectors = {}
    for i, embedding in enumerate(Vecs):
        vectors[i] = embedding
    return vectors
