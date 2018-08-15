import copy
import logging
import random
import sys

import numpy as np

from ..core import UserAct, SystemAct
from ..util import find_slot_with_key, b64_to_img, build_slot_dict, sort_slots_with_key

logger = logging.getLogger(__name__)


def UserPortal(user_config):
    user_type = user_config["USER"]
    dice_threshold = float(user_config["DICE_THRESHOLD"])
    select_prob = float(user_config["SELECT_PROB"])
    patience = int(user_config["PATIENCE"])
    return builder(user_type)(select_prob, dice_threshold, patience)


def build_user_act(da, intent=None, slots=None):
    user_act = {}
    user_act['dialogue_act'] = build_slot_dict('dialogue_act', da)
    if intent is not None:
        user_act['intent'] = build_slot_dict("intent", intent)
    if slots is not None:
        user_act['slots'] = slots
    return user_act


class AgendaBasedUserSimulator(object):
    """
    An implementation of an agenda based user simulator
    for image editing requests.
    """

    def __init__(self, select_prob, dice_threshold, patience):
        """
        Initialize user simulator with configuration
        """
        self.select_prob = select_prob
        self.dice_threshold = dice_threshold
        self.patience = patience

    def load_agenda(self, agenda):
        """
        Args:
            agenda (list): list of goals
        """
        self.agenda = agenda

    def process_agenda(self, agenda):
        raise NotImplementedError

    def get_current_goal(self):
        """
        Gets the top goal of the agenda
        Returns:
            goal (dict)
        """
        if self.agenda is None or len(self.agenda) == 0:
            return None
        return self.agenda[0]

    def reset(self):
        self.agenda = None
        self.observation = {}
        self.turn_id = 0

    def observe(self, observation):
        """
        Updates observation
        """
        self.observation.update(observation)

    def act(self):
        """
        Perform actions according to the current agenda
        """
        assert self.agenda is None or len(self.agenda) > 0

        user_acts = []

        # The default action for the user is to inform the agenda
        default_system_acts = [
            {'dialogue_act': build_slot_dict('dialogue_act', SystemAct.GREETING)}]

        # We needs system_acts
        system_acts = self.observation.get('system_acts', default_system_acts)
        for sys_act in system_acts:
            # We can do a for loop here of the system acts
            # because there can only be one system act that
            # requires a response from the user
            sys_dialogue_act = sys_act['dialogue_act']['value']
            if sys_dialogue_act == SystemAct.GREETING:
                user_act = self.act_inform_goal()
            elif sys_dialogue_act == SystemAct.REQUEST:
                request_slots = sys_act["slots"]
                user_act = self.act_inform_request(request_slots)

            elif sys_dialogue_act == SystemAct.CONFIRM:
                confirm_slots = sys_act["slots"]
                user_act = self.act_confirm(confirm_slots)

            elif sys_dialogue_act in SystemAct.query_acts():
                user_act = {}
                user_act['dialogue_act'] = build_slot_dict(
                    "dialogue_act", UserAct.WAIT)

            elif sys_dialogue_act == SystemAct.EXECUTE:
                exec_intent = sys_act.get("intent", None)
                exec_slots = sys_act.get("slots", list())
                success = self.check_execution(exec_intent, exec_slots)
                print("Execution result", success)
                if not success:
                    user_act = {}
                    user_act["dialogue_act"] = build_slot_dict(
                        'dialogue_act', 'inform')
                    user_act["intent"] = build_slot_dict("intent", "undo")
                else:
                    self.agenda.pop(0)
                    user_act = self.act_inform_goal()
            user_acts.append(user_act)

        # Update turn_id
        self.turn_id += 1

        # Check episode_done based on goal
        curr_goal = self.get_current_goal()
        episode_done = curr_goal["intent"]["value"] == "close" or self.turn_id >= self.patience

        # Build return object
        user_act = {}
        user_act['user_acts'] = user_acts
        user_act['user_utterance'] = self.template_nlg(user_acts)
        user_act['episode_done'] = episode_done
        return user_act

    def act_inform_goal(self):
        """
        Informs the current goal, which is the top of the agenda
        Args:
            slots (list): list of slot names(string) that needs to be informed
        Returns:
            user_act (dict): user act object
        """
        # Get current goal
        goal = self.get_current_goal()

        # user_act
        user_act = copy.deepcopy(goal)
        user_act["dialogue_act"] = build_slot_dict(
            "dialogue_act", UserAct.INFORM)
        goal_slots = user_act.get("slots", list())  # Could be empty
        user_slots = filter(
            lambda slot: slot['slot'] != 'object_mask_str', goal_slots)
        user_slots = list(user_slots)
        user_act["slots"] = user_slots
        return user_act

    def act_inform_request(self, request_slots):
        """
        Inform respond to system request
        Args:
            request_slots (list): list of slot dict 
        Returns:
            user_act (dict)
        """
        req_slot = request_slots[0]
        req_name = req_slot['slot']
        goal_slots = self.get_current_goal()['slots']

        if req_name == "object_mask_id":
            # Compare with all the provided
            print("requesting object_mask_id")

            mask_strs = req_slot['value']
            goal_mask_str = find_slot_with_key(
                'object_mask_str', goal_slots)['value']

            dice_scores = [self.compute_dice(
                mask_str, goal_mask_str) for mask_str in mask_strs]

            max_score = max(dice_scores)
            max_index = dice_scores.index(max_score)

            inform_slot = build_slot_dict('object_mask_id')
            if max_score >= self.dice_threshold:
                inform_slot['value'] = max_index
            else:
                inform_slot['value'] = -1
        else:
            inform_slot = find_slot_with_key(req_name, goal_slots)

        if inform_slot is None:
            # Inform goal by default
            return self.act_inform_goal()

        user_act = {}
        user_act['dialogue_act'] = build_slot_dict(
            'dialogue_act', UserAct.INFORM)
        user_act['slots'] = [inform_slot]
        return user_act

    def act_confirm(self, confirm_slots):
        assert len(confirm_slots) == 1
        confirm_slot = confirm_slots[0]
        goal_slots = self.get_current_goal()["slots"]
        target_slot = find_slot_with_key(confirm_slot['slot'], goal_slots)

        if target_slot is None:
            # Is adjust_value 50?
            # Select the dog
            return self.act_inform_goal()

        # Check value
        if confirm_slot['slot'] == "object_mask_str":
            # Calculate dice_score
            mask_str = confirm_slot['value']
            goal_mask_str_slot = find_slot_with_key(
                'object_mask_str', goal_slots)
            goal_mask_str = goal_mask_str_slot['value']
            dice_score = self.compute_dice(mask_str, goal_mask_str)
            same = dice_score >= self.dice_threshold
        else:
            same = confirm_slot['value'] == target_slot['value']

        if same:
            da = UserAct.AFFIRM
        else:
            da = UserAct.NEGATE

        user_act = {}
        user_act['dialogue_act'] = build_slot_dict('dialogue_act', da)
        return user_act

    def check_execution(self, execute_intent, execute_slots):
        """
        Check if system execution result matches current goal
        Returns:
            success (bool): True if success else False
        """
        goal = self.get_current_goal()

        # Check intents
        if execute_intent['value'] != goal['intent']['value']:
            return False

        # Check value of goal_slots
        for target_slot in goal['slots']:
            slot_name = target_slot['slot']
            target_value = target_slot['value']

            if slot_name == "object":  # Skip this slot in object_goal
                continue

            slot = find_slot_with_key(slot_name, execute_slots)
            if slot is None:
                return False
            slot_value = slot['value']

            if slot_name == "object_mask_str":
                dice_score = self.compute_dice(slot_value, target_value)
                if dice_score < self.dice_threshold:
                    return False
            else:
                if slot['value'] != target_value:
                    return False

        return True

    def compute_dice(self, mask_str, goal_mask_str):
        """
        The eyes of the user
        Computes the dice metric between the mask_str and the target_mask_str
        Args:
            mask_str (str): b64 image string
            target_mask_str (str): b64 image string
        Returns:
            dice (float): dice metric
        """
        mask = b64_to_img(mask_str)
        goal_mask = b64_to_img(goal_mask_str)
        assert mask.shape == goal_mask.shape, "Weird shape"
        dice = 2 * (mask & goal_mask).sum() / (mask.sum() + goal_mask.sum())
        return dice

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
                    if slot["slot"] == "object_mask_str":
                        slot_msg = slot["slot"] + "=" + slot["value"][:5]
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
