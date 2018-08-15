import logging
import sys

import numpy as np

from ..core import SystemAct
from ..util import load_from_json, slots_to_args, build_slot_dict

logger = logging.getLogger(__name__)


def build_sys_act(dialogue_act, intent=None, slots=None):
    sys_act = {}
    sys_act['dialogue_act'] = build_slot_dict("dialogue_act", dialogue_act)
    if intent is not None:
        sys_act['intent'] = build_slot_dict("intent", intent)
    if slots is not None:
        sys_act['slots'] = slots
    return sys_act


class BasePolicy(object):
    """
    Base class for all policies
    Attributes:
        state_size (int):
        action_size (int):
        action_map (dict): map action index to system actions
    """

    def __init__(self, state_size, ontology_json):
        self.state_size = state_size
        self.build_action_map(ontology_json)
        self.action_size = len(self.action_map)

    def build_action_map(self, ontology_json):
        """
        """
        action_map = {}

        # Request
        for slot in ontology_json["slots"]:
            action_info = {
                'dialogue_act': SystemAct.REQUEST,
                'slot': slot["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Confirm
        for slot in ontology_json["slots"]:
            action_info = {
                'dialogue_act': SystemAct.CONFIRM,
                'slot': slot["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Query
        action_idx = len(action_map)
        action_map[action_idx] = {
            'dialogue_act': SystemAct.QUERY,
            'slot': 'object'
        }

        # Execute
        for intent in ontology_json["intents"]:
            action_info = {
                'dialogue_act': SystemAct.EXECUTE,
                'intent': intent["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        self.action_map = action_map
        return self.action_map

    def next_action(self, state):
        """
        Args:
            state (object): state of the system
        Returns:
            sys_act (dict): one system_action
        """
        raise NotImplementedError

    def step(self, state):
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """
    Picks an action randomly
    """

    def next_action(self, state):
        raise NotImplementedError

    def step(self, state=None):
        """
        Much as a gym agent step, take a vector as state input
        Args:
            state (list): list of floats
        Returns:
            action (int): action index
        """
        return np.random.randint(0, self.action_size)


class RulePolicy(BasePolicy):
    """
    Rule based policy using sysintent pulling
    """

    def next_action(self, state):
        """
        A simple rule-based policy
        Returns:
            sys_act (list): list of sys_acts
        """
        sysintent = state.pull()

        if len(sysintent.confirm_slots):
            da = SystemAct.CONFIRM
            intent = da
            slots = sysintent.confirm_slots[:1].copy()  # confirm 1 at a time
        elif len(sysintent.request_slots):
            da = SystemAct.REQUEST
            intent = da
            slots = sysintent.request_slots[:1].copy()  # request 1 at a time

        elif len(sysintent.query_slots):
            da = SystemAct.QUERY
            intent = da
            slots = sysintent.query_slots.copy()

        else:
            da = SystemAct.EXECUTE
            intent = state.get_slot('intent').get_max_value()
            slots = sysintent.execute_slots.copy()

        sys_act = build_sys_act(da, intent, slots)
        return sys_act


def builder(string):
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError as e:
        print(e)
        logger.error("Unknown policy: {}".format(string))
        return None
