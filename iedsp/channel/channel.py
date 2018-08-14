import copy
import sys

import numpy as np

from ..core import UserAct
from ..util import load_from_json


def ChannelPortal(config):
    ontology_file = config["DEFAULT"]["ONTOLOGY_FILE"]

    channel_config = config["CHANNEL"]
    channel_type = channel_config["CHANNEL"]
    speech_conf_mean = float(channel_config["SPEECH_CONF_MEAN"])
    speech_conf_std = float(channel_config["SPEECH_CONF_STD"])
    return builder(channel_type)(ontology_file, speech_conf_mean, speech_conf_std)


class MultimodalChannel(object):
    """
    A channel that simulates the photohsop interface & speech recognition input
    Assign confidence scores based on the slots provided.

    """

    def __init__(self, ontology_file, speech_conf_mean, speech_conf_std):
        """
        Loads configurations
        """
        self.ontology_json = load_from_json(ontology_file)
        self.speech_conf_mean = speech_conf_mean
        self.speech_conf_std = speech_conf_std

        # Process
        self._process_ontology()

    def _process_ontology(self):

        intents = self.ontology_json["intents"]

        self.speech_intents = []
        self.photoshop_intents = []
        for intent in intents:
            intent_name = intent["name"]
            if intent.get("speech", False):
                self.speech_intents.append(intent_name)
            else:
                self.photoshop_intents.append(intent_name)

        slots = self.ontology_json["slots"]

        self.speech_slots = []
        self.photoshop_slots = []

        for slot in slots:
            slot_name = slot["name"]
            if slot["node"] == "BeliefNode":
                self.speech_slots.append(slot_name)
            else:
                self.photoshop_slots.append(slot_name)

    def reset(self):
        """ 
        We don't need to do anything
        """
        pass

    def observe(self, observation):
        """
        Observes user act
        """
        self.observation = observation

    def act(self):
        """
        Iterate through confidence scores from the user and assign confidence scores
        Returns:
            channel_act (dict): user_act with channel confidence scores
        """
        channel_act = copy.deepcopy(self.observation)
        for user_act in channel_act['user_acts']:
            # Assign confidence scores
            user_act["dialogue_act"]["conf"] = self.generate_confidence()

            if "intent" in user_act:
                intent_name = user_act["intent"]["value"]
                if intent_name in self.photoshop_intents:
                    user_act["intent"]["conf"] = 1.
                else:
                    user_act["intent"]["conf"] = self.generate_confidence()

            for slot_dict in user_act.get('slots', list()):
                slot_name = slot_dict["slot"]
                if slot_name in self.photoshop_slots:
                    slot_dict["conf"] = 1.
                else:
                    slot_dict['conf'] = self.generate_confidence()

        return channel_act

    def corrupt(self, slot_dict):
        """
        Modify the slot values according to the assigned confidence
        """
        pass

    def generate_confidence(self):
        """
        Samples a confidence scores with mean and std and to 2 floating points
        Args:
            sample (bool): 
        Returns:
            conf_score (float)
        """
        conf_score = np.random.normal(
            self.speech_conf_mean, self.speech_conf_std)
        conf_score = round(conf_score, 2)
        conf_score = max(conf_score, 0.0)  # >= 0.
        conf_score = min(conf_score, 1.0)  # <= 1.
        return conf_score


def builder(string):
    """
    Gets node class with string
    """
    return getattr(sys.modules[__name__], string)
