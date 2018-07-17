"""
    Class to simulate speech recognition errors
"""
import copy

import numpy as np


class ErrorChannel(object):
    def __init__(self, opt=None):
        self.conf_mean = 0.8
        self.conf_std = 0.15

    def reset(self):
        pass

    def observe(self, observation):
        """
            Dive into the user_act object and assign confidence scores
        """
        self.observation = observation

    def act(self):
        user_act = self.observation.get('user_acts')

        conf_observation = copy.deepcopy(self.observation)

        conf_user_act = copy.deepcopy(user_act)

        for user_act in conf_user_act:
            import pdb
            pdb.set_trace()

    def _corrupt(self, slot_value_dict):
        slot_value_dict = copy.deepcopy(slot_value_dict)
        conf_score = np.random.normal(self.conf_mean, self.conf_std)
        conf_score = round(conf_score, 2)
        conf_score = max(conf_score, 0.0)  # >= 0.
        conf_score = min(conf_score, 1.0)  # <= 1.
        slot_value_dict['conf'] = conf_score
        return slot_value_dict


if __name__ == "__main__":
    pass
