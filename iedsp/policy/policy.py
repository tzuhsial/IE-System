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
    Base class for all policies, also records reward

    public methods:
        next_action
        record
    private methods:
        build_action_map
        step
        action_idx_to_sys_act

    Attributes:
        state_size (int):
        action_size (int):
        action_map (dict): map action index to system actions
    """

    def __init__(self, state_size, ontology_json):
        self.state_size = state_size
        self.build_action_map(ontology_json)
        self.action_size = len(self.action_map)

    def reset(self):
        self.rewards = []

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
        # Predict next action index
        state_list = state.to_list()
        action_idx = self.step(state_list)
        sys_act = self.action_idx_to_sys_act(action_idx, state)
        return sys_act

    def step(self, state):
        """
        Override this class for customized policies
        Args:
            state (list): list of float
        returns:
            action (int): action index
        """
        raise NotImplementedError

    def action_idx_to_sys_act(self, action_idx, state):
        """
        Process action_idx into sys_act object
        """
        action_dict = self.action_map[action_idx]
        da = action_dict['dialogue_act']
        if da == SystemAct.REQUEST:
            intent = da
            slot_name = action_dict['slot']
            slot_dict = build_slot_dict(slot_name)
            slots = [slot_dict]
        elif da in [SystemAct.CONFIRM, SystemAct.QUERY]:
            intent = da
            slot_name = action_dict['slot']
            slot_dict = state.get_slot(slot_name).to_json()
            slots = [slot_dict]
        elif da == SystemAct.EXECUTE:
            intent = action_dict['intent']
            intent_node = state.get_intent(intent)

            slots = []
            for child_node in intent_node.children.values():
                slot_dict = child_node.to_json()
                slots.append(slot_dict)
        else:
            raise ValueError("Unknown dialogue act: {}".format(da))

        sys_act = build_sys_act(da, intent, slots)
        return sys_act

    def add_reward(self, reward):
        """
        Records 
        """
        self.rewards.append(reward)


class RandomPolicy(BasePolicy):
    """
    Picks an action randomly
    """

    def step(self, state):
        """
        Much as a gym agent step, take a vector as state input
        Args:
            state (list): list of floats
        Returns:
            action (int): action index
        """
        return np.random.randint(0, self.action_size)


class CommandLinePolicy(BasePolicy):
    """
    Policy interaction via command line
    """

    def step(self, state):
        """
        Display action index and corresponding action descriptions
        and ask for integer input
        """
        cmd_msg_list = []
        for action_idx, action_dict in self.action_map.items():
            da_name = action_dict.get('dialogue_act')
            slot_name = action_dict.get("intent") or action_dict.get("slot")
            action_msg = "Action {} : {} {}".format(
                action_idx, da_name, slot_name)
            cmd_msg_list.append(action_msg)

        cmd_msg = " | ".join(cmd_msg_list)
        action_index = -1

        while action_index < 0 or action_index >= self.action_size:
            print(cmd_msg)
            action_index = int(
                input("[CMDPolicy] Please input an action index: "))

        return action_index


class RulePolicy(BasePolicy):
    """
    Rule based policy using sysintent pulling
    """

    def next_action(self, state):
        """
        A simple rule-based policy
        Args:
            state (object): State defined in state.py
            reward (float): reward resulting from previous action
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
