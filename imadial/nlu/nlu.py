import copy
import os
import re
import sys

import requests
from nltk import edit_distance
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from ..util import build_slot_dict


english_stopwords = stopwords.words('english')
stemmer = PorterStemmer()

integer_pattern = r"-?\d+"


def NLUPortal(nlu_config):
    nlu_type = nlu_config["nlu"]
    uri = nlu_config["uri"]
    return builder(nlu_type)(uri)


class NLIETagger(object):

    def __init__(self, uri):
        self.uri = uri

    def reset(self):
        self.observation = {}

    def observe(self, observation):
        self.observation = observation

    def act(self):
        sentence = self.observation.get("user_utterance", "")

        sentence = sentence.strip().lower()

        if sentence in ["yes", "no"]:
            nlu_act = self.act_confirm(sentence)
        else:
            nlu_act = self.act_inform(sentence)

        user_act = copy.deepcopy(self.observation)
        user_act['user_acts'] = [nlu_act]
        return user_act

    def act_inform(self, sentence):
        data = {"sent": sentence}

        try:
            res = requests.post(self.uri, data=data)
            res.raise_for_status()
            tagged = res.json()

            slots = []

            # Track individual slots
            # Intent
            intent_value = tagged["intent"]
            slot = build_slot_dict('intent', intent_value, 1.0)
            slots.append(slot)

            # Attribute and referring expression
            individual_slots = ["attribute", "refer"]
            for slot_name in individual_slots:
                if len(tagged[slot_name]) > 0:
                    value = tagged[slot_name][0]
                    if slot_name == "refer":
                        slot = build_slot_dict('object', value, 1.0)
                    else:
                        slot = build_slot_dict(slot_name, value, 1.0)
                    slots.append(slot)

            # Get action word positive negative
            value_sign = 1
            negative_action_words = ["decrease", "lower", "reduce", "drop"]
            if len(tagged["action"]) > 0:
                action_word = tagged["action"][0]
                if action_word in negative_action_words:
                    value_sign = -1

            if len(tagged["value"]) > 0:
                value = int(tagged["value"][0])
                if value_sign < 0 and value > 0:
                    value *= -1
                slot = build_slot_dict('adjust_value', value, 1.0)
                slots.append(slot)

        except Exception as e:
            print(e)
            slots = []

        nlu_act = {
            'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
            'slots': slots
        }

        return nlu_act

    def act_confirm(self, sentence):
        if sentence == "yes":
            da = "affirm"
        elif sentence == "no":
            da = "negate"
        else:
            raise ValueError("Unknown confirm sentence: {}".format(sentence))

        nlu_act = {'dialogue_act': {
            "slot": "dialogue_act", "value": da, 'conf': 1.0}}
        return nlu_act


class EditmeTagger(object):
    """
    Calls the nlu on editme for state updates
    """

    def __init__(self, uri):
        self.uri = uri

    def reset(self):
        pass

    def observe(self, observation):
        self.observation = observation

    def act(self):
        sentence = self.observation.get("user_utterance", "")
        sentence = sentence.strip().rstrip(".")
        if sentence.lower() in ["", "undo", "redo", "close"]:
            nlu_act = self.act_inform(sentence)
        elif sentence.lower() in ["yes", "no"]:
            nlu_act = self.act_confirm(sentence)
        else:
            nlu_act = self.act_editme(sentence)

        act = copy.deepcopy(self.observation)
        act['user_acts'] = [nlu_act]
        return act

    def act_inform(self, intent):
        nlu_act = {
            'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
            'slots': []
        }
        if intent != "":
            nlu_act['intent'] = build_slot_dict('intent', intent, 1.0)
        return nlu_act

    def act_confirm(self, sentence):
        if sentence.lower() == "yes":
            da = "affirm"
        elif sentence.lower() == "no":
            da = "negate"
        else:
            raise ValueError("Unknown confirms sentence: {}".format(sentence))

        nlu_act = {'dialogue_act': {
            "slot": "dialogue_act", "value": da, 'conf': 1.0}}
        return nlu_act

    def act_editme(self, sentence):
        data = {"sentence": sentence}

        # Query Editme Tagger
        res = requests.post(self.uri, data=data)
        obj = res.json()

        intent = obj["intent"]
        tags = obj["tags"]
        tokens = obj["tokens"]

        # Write Rules to tailor to our domain
        if sentence.strip().lower() in ["adjust", "undo", "redo", "close"]:
            intent = sentence.strip().lower()

        # Process Tags here
        slots = []
        slot = {}
        # Simple extract the last word of each IOB tag
        # Group tag and words
        prev_iob = None
        prev_label = None

        for word, tag in zip(tokens, tags):
            if tag == 'O':
                continue

            iob, label = tag.split('-')  # B-mask

            if prev_label != None:
                if label != prev_label or iob == 'B':
                    if len(slot):
                        slots.append(slot)
                        slot = {}
            if label not in slot:
                slot[label] = list()
            slot[label].append(word)

            # Append only if you have a new occurence
            prev_label = label
            prev_iob = iob

        if len(slot):
            slots.append(slot)

        # Now we can filter the stuff we want
        updated_slots = []

        for slot in slots:
            slot_type, tokens = list(slot.items())[0]
            word = tokens[-1]

            if word in english_stopwords:
                continue
            if slot_type == "attribute":  # attribute
                s = slot_type
                v = word

                matched = False
                if v.startswith('bright'):
                    v = "brightness"
                elif v.startswith('saturat'):
                    v = "saturation"
                elif v.startswith('light'):
                    v = "lightness"

                if v not in ["brightness", "contrast", "hue", "saturation", "lightness"]:
                    continue

            elif slot_type == "value":  # adjust_value
                if word not in ["more", "less"]:
                    continue
                s = "adjust_value"
                v = {'more': 10, 'less': -10}.get(word)
            elif slot_type == "mask":  # object
                s = "object"
                v = word
            else:
                continue

            slot_dict = {'slot': s, 'value': v, 'conf': 1.0}
            updated_slots.append(slot_dict)

        # Special case: adjust_value, use regex
        matches = re.findall(integer_pattern, sentence)
        if len(matches):
            adjust_value_slot = {
                'slot': 'adjust_value', 'value': int(matches[0]), 'conf': 1.0}
            updated_slots.append(adjust_value_slot)

        intent_slot = {'slot': 'intent', 'value': intent, 'conf': 1.0}
        nlu_act = {
            'dialogue_act': {'slot': 'dialogue_act', 'value': 'inform', 'conf': 1.0},
            'intent': intent_slot,
            'slots': updated_slots
        }
        return nlu_act


def builder(string):
    return getattr(sys.modules[__name__], string)


if __name__ == "__main__":
    nlu = EditmeTagger("http://localhost:2004/tag")

    with open('testing.txt', 'r') as fin:
        next(fin)
        for line in fin.readlines():
            pos, intent = line.strip().split('|')
            pairs = pos.split()
            sentence, tags = zip(*(pair.split("###") for pair in pairs))
            sentence = ' '.join(sentence)
            obsrv = {"user_utterance": sentence}
            print(sentence)
            nlu.observe(obsrv)
            act = nlu.act()
