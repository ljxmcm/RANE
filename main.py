from evaluate import *
from rank import *
from train import *
from mapping import *
import os

# The dominant components of each dataset are different
# A larger value is assigned to topology parameter C_a  if graph structure is more informative than label information.
# Here are the values we empirically found to work well for different datasets:
#cora        C_a = 0.34, C_x = node_size/3,   alpha = 0.1, Micro-f1 = 0.8786
#citeseer  C_a = 0.4,   C_x = node_size/20, alpha = 0.3, Micro-f1 = 0.7608
#wiki         C_a = 0.5,   C_x = node_size/8,   alpha = 0.05, Micro-f1 = 0.802

os.environ['CUDA_VISIBLE_DEVICES'] = "6"       

R_Model = R_Model(edgelist='cora/cora.edgelist', features='cora/cora.features')

#ranking
R_x = R_Model.rank_X()
R_A = R_Model.rank_A(alpha=0.05)
#mapping
C_a = 0.5
C_x = R_Model.node_size/8
T = mapping(vector=R_x, a=1, b=0, c=C_x)*mapping(vector=R_A, a=1, b=1, c=C_a)
T = normalize(T)
#train
RANE = T_Model(T)
                   
#evaluate
micro_max = 0
epoch_max = 0
for i in range(200):
    print('epoch:{}'.format(i))
    RANE.train_one_epoch()
    vectors = RANE.get_embeddings()
    score = evaluation('cora/cora.labels',  vectors, 0.5)
    if score >= micro_max:
        micro_max = score
        epoch_max = i
        best_vectors = vectors
print("max_epoch:{}, max_micro:{}".format(epoch_max, micro_max))
save_embeddings(best_vectors, 'RANE')