import copy
import logging
import random
import sys

import numpy as np

from ..core import UserAct, SystemAct
from ..util import find_slot_with_key, b64_to_img, build_slot_dict, sort_slots_with_key

logger = logging.getLogger(__name__)


def UserPortal(global_config):
    user_config = global_config["USER"]
    user_type = user_config["USER"]
    dice_threshold = float(user_config["DICE_THRESHOLD"])
    patience = int(user_config["PATIENCE"])

    reward_config = global_config["REWARD"]

    return builder(user_type)(dice_threshold, patience)


def build_user_act(da, intent=None, slots=None):
    user_act = {}
    user_act['dialogue_act'] = build_slot_dict('dialogue_act', da)
    if intent is not None:
        user_act['intent'] = build_slot_dict("intent", intent)
    if slots is not None:
        user_act['slots'] = slots
    return user_act


class UserReward(object):
    """
    Defines the reward observed by the agent
    """

    def __init__(self):
        """
        """
        pass

    def get(self, sys_act):
        """
        Reward is defined based on sys_act
        """
        sys_dialogue_act = sys_act['dialogue_act']['value']
        if sys_dialogue_act == SystemAct.REQUEST:
            slots = sys_act['slots']
            if find_slot_with_key('object_mask_str', slots) is not None:
                return -3
        return -1

    def get_final(self, agenda):
        pass


class AgendaBasedUserSimulator(object):
    """
    Agenda based user simulator for image edit requests
    Also contains reward definitions for reinforcement learning
    """

    def __init__(self, dice_threshold, patience):
        """
        Initialize user simulator with configuration
        """
        self.dice_threshold = dice_threshold
        self.patience = patience

    def load_agenda(self, agenda):
        """
        Args:
            agenda (list): list of goals
        """
        self.agenda = agenda.copy()

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

    def get_photoshop_slot(self, slot_name):
        """
        Find desired slot from photoshop
        """
        ps_acts = self.observation.get('photoshop_acts', [{}])
        ps_act = ps_acts[0]
        ps_slots = ps_act.get('slots', list())
        target_slot = find_slot_with_key(slot_name, ps_slots)
        return target_slot

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
        assert len(system_acts) == 1

        sys_act = system_acts[0]

        sys_dialogue_act = sys_act['dialogue_act']['value']

        if sys_dialogue_act == SystemAct.GREETING:
            user_act = self.act_inform_goal()
            reward = 0

        elif sys_dialogue_act == SystemAct.REQUEST:
            request_slots = sys_act["slots"]
            user_act = self.act_inform_request(request_slots)

            if user_act is not None and request_slots[0]['slot'] == "object_mask_str":
                reward = -5
            else:
                reward = -1

        elif sys_dialogue_act == SystemAct.CONFIRM:
            confirm_slots = sys_act["slots"]
            user_act = self.act_confirm(confirm_slots)

            reward = -1

        elif sys_dialogue_act in SystemAct.query_acts():
            user_act = self.act_wait()
            reward = -1

        elif sys_dialogue_act == SystemAct.EXECUTE:

            if not self.get_photoshop_slot('execute_result')['value']:
                # Failed exeuction on Photoshop => doesn't mean anything to the user.
                user_act = None
                reward = -1
            else:
                exec_intent = sys_act.get("intent", None)
                exec_slots = sys_act.get("slots", list())
                success = self.check_execution(exec_intent, exec_slots)
                if not success:
                    print("[user] check action failure")
                    undo_goal = build_user_act('inform', 'undo')
                    self.agenda.insert(0, undo_goal)
                    reward = -10
                else:
                    print("[user] check action success")
                    if len(self.agenda) > 0:
                        self.agenda.pop(0)
                    reward = 10

                user_act = self.act_inform_goal()
        else:
            raise ValueError(
                "Unknown sys_dialogue_act: {}".format(sys_dialogue_act))

        if user_act is None:
            user_act = self.act_inform_goal()

        user_acts.append(user_act)

        # Update turn_id
        self.turn_id += 1

        # Check episode_done based on goal
        episode_done = len(self.agenda) == 0 or self.turn_id >= self.patience

        # Build return object
        user_act = {}
        user_act['user_acts'] = user_acts
        user_act['user_utterance'] = self.template_nlg(user_acts)
        user_act['episode_done'] = episode_done
        user_act['reward'] = reward
        return user_act

    def act_inform_goal(self):
        """
        Informs the current goal, which is the top of the agenda
        Args:
            slots (list): list of slot names(string) that needs to be informed
        Returns:
            user_act (dict): user act object
            reward (float): reward
        """
        # Get current goal
        goal = self.get_current_goal()
        if goal is None:
            return build_user_act(UserAct.BYE)

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
            user_act (dict): None if invalid
        """

        req_name = request_slots[0]['slot']
        goal = self.get_current_goal()
        if req_name == "intent":
            goal_slots = [goal['intent']]
        else:
            goal_slots = goal['slots']

        if req_name == "object_mask_id":
            # Get mask_strs from photoshop
            mask_strs_slot = self.get_photoshop_slot('mask_strs')
            mask_strs = mask_strs_slot['value']

            if len(mask_strs) == 0:
                return None

            goal_mask_str_slot = find_slot_with_key(
                'object_mask_str', goal_slots)
            if goal_mask_str_slot is None:
                return None

            goal_mask_str = goal_mask_str_slot['value']

            dice_scores = [self.compute_dice(
                mask_str, goal_mask_str) for mask_idx, mask_str in mask_strs]

            # Find the one with the maximum dice score
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
            return None

        user_act = build_user_act(
            UserAct.INFORM, UserAct.INFORM, [inform_slot])
        return user_act

    def act_confirm(self, confirm_slots):

        confirm_name = confirm_slots[0]['slot']
        confirm_value = confirm_slots[0].get('value', None)
        # Check value
        if confirm_value is None:
            return None

        goal = self.get_current_goal()
        if confirm_name == "intent":
            goal_slots = [goal['intent']]
        else:
            goal_slots = goal['slots']

        target_slot = find_slot_with_key(confirm_name, goal_slots)

        # What are you confirming?
        if target_slot is None:
            return None

        # Check values
        target_value = target_slot['value']
        if confirm_name == "object_mask_str":
            # Calculate dice_score
            mask_str = confirm_value
            goal_mask_str_slot = find_slot_with_key(
                'object_mask_str', goal_slots)
            goal_mask_str = goal_mask_str_slot['value']
            dice_score = self.compute_dice(mask_str, goal_mask_str)
            same = dice_score >= self.dice_threshold
        else:
            same = confirm_value == target_value

        if same:
            da = UserAct.AFFIRM
        else:
            da = UserAct.NEGATE

        # Yes/No
        user_act = build_user_act(da)
        return user_act

    def act_wait(self):
        user_act = build_user_act(UserAct.WAIT)
        return user_act

    def act_bye(self):
        user_act = build_user_act(UserAct.BYE)
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
        for target_slot in goal.get('slots', list()):
            slot_name = target_slot['slot']
            target_value = target_slot['value']

            if slot_name == "object":  # Skip this slot in object_goal
                continue

            slot = find_slot_with_key(slot_name, execute_slots)
            if slot is None or not slot.get('value'):
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
            elif user_dialogue_act == UserAct.BYE:
                utt = "Bye."
            else:
                raise ValueError(
                    "Unknown user_dialogue_act: {}".format(user_dialogue_act))
            utt_list.append(utt)

        utterance = ' '.join(utt_list)
        return utterance


def builder(string):
    return getattr(sys.modules[__name__], string)
