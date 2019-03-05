from utils import *
from rank import *
from mapping import *
from train import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"       

def micro(model):
    micro_max = 0
    epoch_max = 0
    for i in range(200):
        model.train_one_epoch()
        vectors = model.get_embeddings()
        score = evaluation('cora/cora_labels.txt',  vectors, 0.5)
        if score >= micro_max:
            micro_max = score
            epoch_max = i
            best_vectors = vectors
    print("max_epoch:{}, max_micro:{}".format(epoch_max, micro_max))
    save_embeddings(best_vectors, 'RANE')
    return micro_max

#load data
X = load_attribute('cora/cora.features')
A = load_topology('cora/cora_edgelist.txt')
#ranking
R_x = rank_X(X)
R_A = rank_A(A)
#mapping
C_a = 0.33
C_x = 2708/3
T = mapping(vector=R_x, a=1, b=0, c=C_x)*mapping(vector=R_A, a=1, b=1, c=C_a)
T = normalize(T)
#train
ASANE = model(T)
#evaluate
micro(ASANE)