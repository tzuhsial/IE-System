import copy
import logging
import sys

import numpy as np

from ..core import SystemAct
from ..util import load_from_json, slots_to_args, build_slot_dict, find_slot_with_key

logger = logging.getLogger(__name__)


def build_sys_act(dialogue_act, intent=None, slots=None):
    sys_act = {}
    sys_act['dialogue_act'] = build_slot_dict("dialogue_act", dialogue_act)
    if intent is not None:
        sys_act['intent'] = build_slot_dict("intent", intent)
    if slots is not None:
        sys_act['slots'] = slots
    return sys_act


class ActionMapper(object):
    """
    Maps index to user_act with ontology_json
    """

    def __init__(self, action_config):
        """
        Builds action map here
        """
        # Setup config
        self.config = action_config

        # Build action map
        action_map = {}

        # Request
        for slot in action_config["request"]:
            action_info = {'dialogue_act': SystemAct.REQUEST, 'slot': slot}
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Confirm
        for slot in action_config["confirm"]:
            action_info = {'dialogue_act': SystemAct.CONFIRM, 'slot': slot}
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Query
        for slot in action_config["query"]:
            action_info = {'dialogue_act': SystemAct.QUERY, 'slot': slot}
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Execute
        for intent in action_config["execute"]:
            action_info = {'dialogue_act': SystemAct.EXECUTE, 'intent': intent}
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        self.action_map = action_map

        # Build inverse action map here
        self.inv_map = {}
        self.inv_map[SystemAct.REQUEST] = {}
        self.inv_map[SystemAct.CONFIRM] = {}
        self.inv_map[SystemAct.QUERY] = {}
        self.inv_map[SystemAct.EXECUTE] = {}

        for action_idx, action_info in self.action_map.items():
            da = action_info['dialogue_act']
            if da != SystemAct.EXECUTE:
                slot_name = action_info['slot']
                self.inv_map[da][slot_name] = action_idx
            else:
                intent_name = action_info['intent']
                self.inv_map[da][intent_name] = action_idx

    def size(self):
        return len(self.action_map)

    def __call__(self, action_idx, state):
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

    def find_action_idx(self, da, intent=None, slot=None):
        key = intent or slot
        return self.inv_map[da][key]

    def sys_act_to_action_idx(self, sys_act):
        da = sys_act["dialogue_act"]['value']
        intent = sys_act["intent"]
        slots = sys_act["slots"]

        if da != SystemAct.EXECUTE:
            key = slots[0]['slot']
        else:
            key = intent['value']
        action_idx = self.inv_map[da][key]
        return action_idx


class BasePolicy(object):
    """
    public methods:
        next_action

    private methods:
        build_action_map
        step
        action_idx_to_sys_act

    Attributes:
        state_size (int):
        action_size (int):
        action_map (dict): map action index to system actions
    """

    def __init__(self, policy_config, action_mapper, **kwargs):
        self.config = policy_config
        self.state_size = policy_config["state_size"]
        self.action_size = policy_config["action_size"]
        self.action_mapper = action_mapper

    def reset(self):
        self.rewards = []
        self.previous_state = None
        self.previous_action = None
        self.state = None
        self.action = None
        self.reward = None
        self.episode_done = None

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
        sys_act = self.action_mapper(action_idx, state)

        # Feature size
        self.previous_state = self.state
        self.previous_action = self.action
        self.state = state_list
        self.action = action_idx
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

    def to_json(self):
        return {}

    def from_json(self, obj):
        pass


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
        if len(sysintent.query_slots):  # Query First
            da = SystemAct.QUERY
            intent = da
            slots = sysintent.query_slots.copy()

        elif len(sysintent.confirm_slots):
            # confirm one at a time, order does not matter
            da = SystemAct.CONFIRM
            intent = da

            slots = sysintent.confirm_slots[:1].copy()

        elif len(sysintent.request_slots):
            da = SystemAct.REQUEST
            intent = da

            order = {
                'object': 1,
                'attribute': 2,
                'adjust_value': 3,
                'intent': 4
            }
            sysintent.request_slots = sorted(
                sysintent.request_slots, key=lambda x: order[x['slot']])
            slots = sysintent.request_slots[:1].copy()  # request 1 at a time

        else:
            da = SystemAct.EXECUTE
            intent = state.get_slot('intent').get_max_value()
            slots = sysintent.execute_slots.copy()

        sys_act = build_sys_act(da, intent, slots)

        action_idx = self.action_mapper.sys_act_to_action_idx(sys_act)

        # Find action index from sys_act
        state_list = state.to_list()

        self.previous_state = self.state
        self.previous_action = self.action
        self.state = state_list
        self.action = action_idx

        return sys_act


def builder(string):
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError as e:
        print(e)
        logger.error("Unknown policy: {}".format(string))
        return None
