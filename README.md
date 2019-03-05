# RANE
Rank based Attributed Network Embedding

## Requirements

The codebase is implemented in Python 3.5.2. package versions used for development are just below.

```
numpy        1.15.4  
sklearn      0.19.0  
tensorflow   1.12.0  
```

## Input 
Each dataset contains 3 files: edgelist, features and labels  

The supported topology input format is an edgelist:
```
node_size
node_1 node_2
node_1 node_3
...
node_n node_1
```
The supported attribute input format is as follow:
```
node feature_1 feature_2 ... feature_n
```
The label format is as follow:
```
node label
```
## Run
```
python main.py
```
The default dataset is Cora.
