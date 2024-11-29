import pickle
import numpy as np
import os

def pickle_data(data,save_fp):
    """ serialize data into pickle
    Args:
        data: posterior distribution
        save_fp (str): save serialized object into fp
    """
    save_fp = os.path.splitext(save_fp)[0]
    with open(f'{save_fp}.pkl','wb') as f:
        pickle.dump(data,f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(fp):
    with open(fp,'rb') as f:
        data = pickle.load(f)
    return data

def logistic(x, beta, alpha):
    """ logistic curve
    """
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))