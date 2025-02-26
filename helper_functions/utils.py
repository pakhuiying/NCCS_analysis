import pickle
import numpy as np
import os
import json

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
    """
    Args:
        fp (str): filepath to pickle object
    Returns:
        pkl: serialised object
    """
    with open(fp,'rb') as f:
        data = pickle.load(f)
    return data

def json_data(data, save_fp):
    """ export data as json format
    Args:
        data (dict): dictionary to be encoded as JSON
        save_fp (str): save json object into fp
    """
    save_fp = os.path.splitext(save_fp)[0]
    with open(f'{save_fp}.json', 'w') as f:
        json.dump(data, f)

def load_json(fp):
    """
    Args:
        fp (str): filepath to convert JSON object to dict
    Returns:
        dict: dictionary
    """
    with open(fp) as f:
        data = json.load(f)
    return data

def load_txt(fp):
    """ 
    Args:
        fp (str): filepath to convert line-by-line txt file into a list
    """
    with open(fp) as f:
        lines = [line.rstrip() for line in f]
    return lines

def logistic(x, beta, alpha):
    """ logistic curve
    """
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))