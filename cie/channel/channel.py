import copy
import sys

import numpy as np

from ..core import UserAct
from .. import util


def ChannelPortal(channel_config):
    channel_type = channel_config["channel"]
    args = {
        "ontology_json": util.load_from_json(channel_config["ontology"]),
        "speech_conf_mean": float(channel_config["speech_conf_mean"]),
        "speech_conf_std": float(channel_config["speech_conf_std"])
    }
    return builder(channel_type)(**args)


class MultimodalChannel(object):
    """
    A channel that simulates the photohsop interface & speech recognition input
    Assign confidence scores based on the slots provided.
    """

    def __init__(self, ontology_json, speech_conf_mean, speech_conf_std):
        """
        Loads configurations
        """
        self.ontology_json = ontology_json
        self.speech_conf_mean = speech_conf_mean
        self.speech_conf_std = speech_conf_std

        # Process
        self._process_ontology()

    def _process_ontology(self):

        self.intents = {}
        for intent in self.ontology_json["intents"]:
            self.intents[intent['name']] = intent

        self.slots = {}
        for slot in self.ontology_json["slots"]:
            self.slots[slot['name']] = slot

    def reset(self):
        """ 
        We don't need to do anything here 
        """
        pass

    def observe(self, observation):
        """
        Observes user act
        """
        self.observation = observation

    def act(self):
        """
        Corrupt the user_actions
        Iterate through confidence scores from the user and assign confidence scores

        Returns:
            channel_act (dict): user_act with channel confidence scores
        """
        channel_act = copy.deepcopy(self.observation)

        for user_act in channel_act['user_acts']:
            # Dialogue Act
            da_conf = self.generate_confidence()
            da_value = user_act["dialogue_act"]["value"]

            if np.random.random() > da_conf:
                if da_value == UserAct.AFFIRM:
                    da_value = UserAct.NEGATE
                elif da_value == UserAct.NEGATE:
                    da_value == UserAct.AFFIRM
                else:
                    pass

            user_act["dialogue_act"]["value"] = da_value
            user_act["dialogue_act"]["conf"] = self.generate_confidence()

            # Intent
            if "intent" in user_act:
                intent_value = user_act["intent"]["value"]
                if self.intents[intent_value].get("speech", False):
                    intent_conf = 1.
                else:
                    intent_conf = self.generate_confidence()

                intent_possible_values = self.slots["intent"][
                    "possible_values"].copy()

                if np.random.random() > intent_conf:
                    intent_possible_values.remove(intent_value)
                    intent_value = np.random.choice(intent_possible_values)

                user_act['intent']['value'] = intent_value
                user_act['intent']['conf'] = intent_conf

            # Slot Values
            for slot_dict in user_act.get('slots', list()):
                slot_name = slot_dict["slot"]
                slot_value = slot_dict["value"]

                if self.slots[slot_name]["node"] != "BeliefNode":
                    slot_conf = 1.0
                else:
                    slot_conf = self.generate_confidence()

                slot_possible_values = self.slots[slot_name].get(
                    "possible_values")

                if slot_possible_values is None:
                    slot_possible_values = list()

                slot_possible_values = slot_possible_values.copy()
                if len(slot_possible_values) and np.random.random() > slot_conf:
                    slot_possible_values.remove(slot_value)
                    slot_value = np.random.choice(slot_possible_values)

                slot_dict['conf'] = slot_conf

        channel_act["channel_utterance"] = self.template_nlg(
            channel_act['user_acts'])
        return channel_act

    def generate_confidence(self):
        """
        Samples a confidence scores with mean and std and to 2 floating points
        Args:
            sample (bool): 
        Returns:
            conf_score (float)
        """
        conf_score = np.random.normal(self.speech_conf_mean,
                                      self.speech_conf_std)
        conf_score = round(conf_score, 2)
        conf_score = max(conf_score, 0.0)  # >= 0.
        conf_score = min(conf_score, 1.0)  # <= 1.
        return conf_score

    def template_nlg(self, user_acts):
        """
        Converts user_acts into tempplate utterances
        Args:
            user_acts (list): list of user_act
        Returns:
            user_utterance (str): the template utterance
        """
        utt_list = []
        for user_act in user_acts:
            user_dialogue_act = user_act['dialogue_act']['value']
            if user_dialogue_act == UserAct.INFORM:
                # Template based NLG based on intent
                intent_slot = user_act.get('intent', None)
                slots = [intent_slot] if intent_slot is not None else []
                slots += user_act.get("slots", list())
                slot_list = []
                for slot in slots:
                    if slot["slot"] in [
                            "object_mask_str", "gesture_click",
                            "original_b64_img_str"
                    ]:
                        slot_msg = slot["slot"] + "=" + slot["value"][:5]
                    elif slot["slot"] == "mask_strs":
                        slot_msg = slot["slot"] + "=" + len(slot["value"])
                    else:
                        slot_msg = slot["slot"] + "=" + str(slot["value"])
                    slot_list.append(slot_msg)
                utt = "I want " + ', '.join(slot_list) + "."
            elif user_dialogue_act == UserAct.AFFIRM:
                utt = "Yes."
            elif user_dialogue_act == UserAct.NEGATE:
                utt = "No."
            elif user_dialogue_act == UserAct.WAIT:
                utt = "(Waiting...)"
            elif user_dialogue_act == UserAct.BYE:
                utt = "Bye."
            else:
                raise ValueError(
                    "Unknown user_dialogue_act: {}".format(user_dialogue_act))
            utt_list.append(utt)

        utterance = ' '.join(utt_list)
        return utterance


def builder(string):
    """
    Gets node class with string
    """
    return getattr(sys.modules[__name__], string)
