"""
    Class to simulate speech recognition errors
"""
import copy

import numpy as np

from ..core import UserAct


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
        for user_act in channel_act['user_acts']:
            # Sample a confidence for dialogue_act
            user_dialogue_act = user_act['dialogue_act']['value']
            # Sample or not depends on user_dialogue_act
            sample = not user_dialogue_act in UserAct.photoshop_acts()
            user_act['dialogue_act']['conf'] = self.generate_confidence(sample)
            #user_act['dialogue_act']['conf'] = 1.
            for slot_dict in user_act.get('slots', list()):
                slot_dict['conf'] = self.generate_confidence(sample)

        return channel_act

    def corrupt(self, slot_dict, channel_conf):
        """
        Corrupts the slot_dict by assigning weird values
        """
        pass

    def generate_confidence(self, sample=True):
        """
        if sample: 
            Samples a confidence scores with mean and std and to 2 floating points
        else:
            returns a confidence score of 1.
        Args:
            sample (bool): 
        Returns:
            conf_score (float)
        """
        if sample:
            conf_score = np.random.normal(self.conf_mean, self.conf_std)
            conf_score = round(conf_score, 2)
            conf_score = max(conf_score, 0.0)  # >= 0.
            conf_score = min(conf_score, 1.0)  # <= 1.
        else:
            conf_score = 1.0
        return conf_score


if __name__ == "__main__":
    pass
