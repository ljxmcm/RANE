import numpy as np

def mapping(vector, a, b, c):
    return a*np.exp(-((vector-b)*(vector-b))/(2*np.power(c,2)))
