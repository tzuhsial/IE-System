from ..util import save_to_pickle


class EvaluationManager(object):
    """
    An evaluation manager
    Attributes: 
        history (list): tuple of (epoch(int), rewards(list), losses(list)
    """

    def __init__(self):

        self.history = {
            "train": {
                "episode": list(),
                "rewards": list(),
                "turns": list(),
                "losses": list()
            },
            "test": {
                "episode": list(),
                "rewards": list(),
                "turns": list(),
                "losses": list()
            }
        }

        self.rewards = []
        self.turns = []
        self.losses = []

    def record(self, ):
        self.rewards.append(reward)
        self.losses.append(loss)

    def record(self, epoch=None, name="train"):
        """
        Args:
            epoch (int): the epoch to be recorded
            name (str): 
        """
        if epoch is None:
            epoch = len(self.history[name]["epoch"])+1

        self.history[name]["epoch"].append(epoch)
        self.history[name]["rewards"].append(self.rewards.copy())
        self.history[name]["losses"].append(self.losses.copy())

    def save(self, path):
        util.save_to_pickle(self.history, path)
