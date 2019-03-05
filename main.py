from evaluate import *
from rank import *
from train import *
from mapping import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"       

R_Model = R_Model(edgelist='cora/cora.edgelist', features='cora/cora.features')

#ranking
R_x = R_Model.rank_X()
R_A = R_Model.rank_A(alpha=0.1)
#mapping
C_a = 0.34
C_x = R_Model.node_size/3
T = mapping(vector=R_x, a=1, b=0, c=C_x)*mapping(vector=R_A, a=1, b=1, c=C_a)
T = normalize(T)
#train
RANE = T_Model(T)
                   
#evaluate
micro_max = 0
epoch_max = 0
for i in range(200):
    RANE.train_one_epoch()
    vectors = RANE.get_embeddings()
    score = evaluation('cora/cora.labels',  vectors, 0.5)
    if score >= micro_max:
        micro_max = score
        epoch_max = i
        best_vectors = vectors
print("max_epoch:{}, max_micro:{}".format(epoch_max, micro_max))
save_embeddings(best_vectors, 'RANE')
