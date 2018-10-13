import copy
import re
import sys

import requests
from nltk import edit_distance
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

english_stopwords = stopwords.words('english')
stemmer = PorterStemmer()

integer_pattern = r"-?\d+"


def TrackerPortal(tracker_config):
    tracker_type = tracker_config["tracker"]
    uri = tracker_config["uri"]
    return builder(tracker_type)(uri)


class EditmeTagger(object):
    """
    Calls the tracker on editme for state updates
    """

    def __init__(self, uri):
        self.uri = uri

    def reset(self):
        pass

    def observe(self, observation):
        self.observation = observation

    def act(self):
        sentence = self.observation.get("user_utterance", "")

        if sentence in ["yes", "no"]:
            tracker_act = self.act_confirm(sentence)
        else:
            tracker_act = self.act_editme(sentence)

        act = copy.deepcopy(self.observation)
        act['user_acts'] = [tracker_act]
        return act

    def act_confirm(self, sentence):
        if sentence == "yes":
            da = "affirm"
        elif sentence == "no":
            da = "negate"

        tracker_act = {'dialogue_act': {
            "slot": "dialogue_act", "value": da, 'conf': 1.0}}
        return tracker_act

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
        tracker_act = {
            'dialogue_act': {'slot': 'dialogue_act', 'value': 'inform', 'conf': 1.0},
            'intent': intent_slot,
            'slots': updated_slots
        }
        return tracker_act


def builder(string):
    return getattr(sys.modules[__name__], string)


if __name__ == "__main__":
    tracker = EditmeTagger("http://localhost:2004/tag")

    with open('testing.txt', 'r') as fin:
        next(fin)
        for line in fin.readlines():
            pos, intent = line.strip().split('|')
            pairs = pos.split()
            sentence, tags = zip(*(pair.split("###") for pair in pairs))
            sentence = ' '.join(sentence)
            obsrv = {"user_utterance": sentence}
            print(sentence)
            tracker.observe(obsrv)
            act = tracker.act()
            pass
