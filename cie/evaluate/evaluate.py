import numpy as np

from .. import util


def print_mean_std(name, seq):
    seq_mean = np.mean(seq)
    seq_std = np.std(seq)
    print(name, "{}(+/-{})".format(seq_mean, seq_std))

class EvaluationManager(object):
    """
    An evaluation manager
    Attributes: 
        history (list): tuple of (epoch(int), rewards(list), losses(list)
    """

    def __init__(self):
        self.history = {}

    def add_summary(self, epoch, name, summary):
        if epoch not in self.history:
            self.history[epoch] = {}
        
        self.history[epoch][name] = summary

    def pprint_summary(self, summary):
        for name, seq in summary.items():
            print_mean_std(name, seq)

    def save(self, path):
        util.save_to_pickle(self.history, path)
