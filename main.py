from evaluate import *
from rank import *
from train import *
from mapping import *
import os

# The dominant components of each dataset are different
# A larger value is assigned to topology parameter C_a  if graph structure is more informative than label information.
# Here are the values we empirically found to work well for different datasets:
#cora        C_a = 0.34,  C_x = node_size/3,   alpha = 0.1
#citeseer    C_a = 0.4,   C_x = node_size/20,  alpha = 0.3
#wiki        C_a = 0.5,   C_x = node_size/8,   alpha = 0.05

#os.environ['CUDA_VISIBLE_DEVICES'] = "6"       
C_a = 0.34
C_x = R_Model.node_size/3
alpha = 0.1
R_Model = R_Model(edgelist='cora/cora.edgelist', features='cora/cora.features')

#ranking
R_X = R_Model.rank_X()
R_A = R_Model.rank_A(alpha=alpha)
#mapping
T = mapping(vector=R_X, a=1, b=0, c=C_x)*mapping(vector=R_A, a=1, b=1, c=C_a)
T = normalize(T)
#training
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
