"""
    Class to simulate speech recognition errors
"""
import copy

import numpy as np

from ..dialogueacts import UserAct


class ConfChannel(object):
    """
    A channel that simulates the errors of speech recognition 
    and Photoshop interaction by assigning confidence scores.
    Currently assigns confidence scores for now.
    TODO:
    1. Error accordiing to confidence
    """

    def __init__(self, config):
        """
        Loads configurations
        """
        self.conf_mean = float(config['CONF_MEAN'])
        self.conf_std = float(config['CONF_STD'])

    def reset(self):
        """ 
        Resets
        """
        pass

    def observe(self, observation):
        """
        Observes user act
        """
        self.observation = observation

    def act(self):
        """
        Receives observation from the user and assign confidence scores
        Returns:
            channel_act (dict): user_act with channel confidence scores
        """
        channel_act = copy.deepcopy(self.observation)

        # Assign confidence scores
        for user_act in channel_act['user_acts']:
            if user_act['dialogue_act'] in [UserAct.OPEN, UserAct.LOAD, UserAct.TOOL_SELECT, UserAct.CLOSE]:
                # There can be no error. Since the user directly interacts with Photoshop
                channel_conf = 1.
            else:
                channel_conf = self.sample_conf()
                channel_conf = 1.

            for slot_dict in user_act.get('slots', list()):
                slot_dict['conf'] = channel_conf

        return channel_act

    def sample_conf(self):
        """
        Samples a confidence scores from Gaussian with mean and std
        Returns:
            conf_score (float)
        """
        conf_score = np.random.normal(self.conf_mean, self.conf_std)
        conf_score = round(conf_score, 2)
        conf_score = max(conf_score, 0.0)  # >= 0.
        conf_score = min(conf_score, 1.0)  # <= 1.
        return conf_score


if __name__ == "__main__":
    pass
