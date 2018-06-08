import pickle

def save_to_pickle(obj, filepath):
    with open(filepath,'wb') as fout:
        pickle.dump(obj, fout)
    
def load_from_pickle(filepath):
    with open(filepath, 'rb') as fin:
        obj = pickle.load(fin)
    return obj
