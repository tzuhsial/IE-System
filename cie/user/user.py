import copy
import itertools
import logging
import random
import sys

import numpy as np

from ..core import UserAct, SystemAct
from ..util import find_slot_with_key, b64_to_img, build_slot_dict, sort_slots_with_key

logger = logging.getLogger(__name__)


def UserPortal(user_config):
    user_type = user_config["user"]
    return builder(user_type)(user_config)


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
    Agenda based user simulator for image edit requests
    Also defines reward model
    """

    def __init__(self, user_config, **kwargs):
        """
        Initialize user simulator with configuration
        """
        self.config = user_config
        self.patience = user_config["patience"]
        self.dice_threshold = user_config["dice_threshold"]
        self.gesture_threshold = user_config["gesture_threshold"]
        self.level = user_config["level"]

    def reset(self):
        self.agenda = None
        self.observation = {}
        self.turn_id = 0

    def load_agenda(self, agenda):
        """
        Args:
            agenda (list): list of goals
        """
        self.agenda = agenda.copy()
        self.agenda_backup = self.agenda.copy()

    def completed_goals(self):
        """
        Returns number of completed goals, excluding "undo, redo"
        """
        remaining_goals = 0
        for g in self.agenda:
            if g['intent']['value'] not in ["undo", "redo"]:
                remaining_goals += 1
        original_goals = len(self.agenda_backup)
        completed_goals = original_goals - remaining_goals
        return completed_goals

    def get_current_goal(self):
        """
        Gets the top goal of the agenda
        Returns:
            goal (dict)
        """
        if self.agenda is None or len(self.agenda) == 0:
            return None
        return self.agenda[0]

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
        act according to observsation
        reward is also defined here
        """
        assert self.agenda is None or len(self.agenda) > 0

        episode_done = False
        reward = 0
        user_acts = []

        # The default action for the user is to inform the agenda
        default_system_acts = [{
            'dialogue_act':
            build_slot_dict('dialogue_act', SystemAct.GREETING)
        }]

        # We needs system_acts
        system_acts = self.observation.get('system_acts', default_system_acts)
        assert len(system_acts) == 1

        sys_act = system_acts[0]
        sys_dialogue_act = sys_act['dialogue_act']['value']

        if sys_dialogue_act == SystemAct.GREETING:
            # Starting state
            user_act = self.act_inform_goal()
            reward = 0

        elif sys_dialogue_act == SystemAct.REQUEST:
            request_slots = sys_act["slots"]
            user_act = self.act_inform_request(request_slots)

            if user_act is not None and request_slots[0]['slot'] == "object_mask_str":
                reward = self.config["request_object_mask_str_penalty"]
            else:
                reward = self.config["turn_penalty"]

        elif sys_dialogue_act == SystemAct.CONFIRM:
            confirm_slots = sys_act["slots"]
            user_act = self.act_confirm(confirm_slots)
            reward = self.config["turn_penalty"]

        elif sys_dialogue_act in SystemAct.query_acts():
            user_act = self.act_wait()
            reward = self.config["turn_penalty"]

        elif sys_dialogue_act == SystemAct.EXECUTE:

            photoshop_execute_slot = self.get_photoshop_slot("execute_result")
            photoshop_execute_result = photoshop_execute_slot['value']

            if photoshop_execute_result:

                exec_intent = sys_act.get("intent", None)
                exec_slots = sys_act.get("slots", list())
                success = self.check_system_execution(exec_intent, exec_slots)

                if success:
                    if len(self.agenda) > 0:
                        self.agenda.pop(0)
                        if len(self.agenda):
                            reward = self.config["success_goal_reward"]
                        else:
                            reward = self.patience - self.turn_id
                    user_act = self.act_inform_goal()
                else:
                    exec_intent_value = exec_intent["value"]
                    if exec_intent_value == "close":
                        episode_done = True
                        reward = -self.patience  # Penalize early closes
                        user_act = self.act_bye()

                    elif exec_intent_value == "undo":
                        # Build redo goal
                        redo_goal = build_user_act('inform', 'redo')
                        self.agenda.insert(0, redo_goal)
                        reward = self.config["failure_goal_penalty"]
                        user_act = self.act_inform_goal()

                    else:  # open, adjust, undo
                        # Build undo goal
                        undo_goal = build_user_act('inform', 'undo')
                        self.agenda.insert(0, undo_goal)
                        reward = self.config["failure_goal_penalty"]
                        user_act = self.act_inform_goal()
            else:
                user_act = None
                reward = self.config["turn_penalty"]
        else:
            raise ValueError(
                "Unknown sys_dialogue_act: {}".format(sys_dialogue_act))

        # Inform goal by default
        if user_act is None:
            user_act = self.act_inform_goal()

        # TODO: customizer user behavior for more than one user acts
        user_acts.append(user_act)

        # Update turn_id
        self.turn_id += 1

        # Episode Done
        if self.turn_id >= self.patience or len(self.agenda) == 0:
            episode_done = True

        if episode_done and len(self.agenda) > 0:
            reward = -self.turn_id

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
        user_act["dialogue_act"] = build_slot_dict("dialogue_act",
                                                   UserAct.INFORM)

        target_slots = user_act.get("slots", list()).copy()  # Could be empty

        if np.random.random() < self.gesture_threshold:
            gesture_slot = find_slot_with_key('gesture_click', target_slots)
            if gesture_slot:
                target_slots.remove(gesture_slot)

        # Users are not assumed to provide stuff in the first place
        filtered_slots = filter(lambda s: s['slot'] != 'object_mask_str',
                                target_slots)

        # Expertise probability dict hard coded here
        expertise = {"expert": 0.0, "intermediate": 0.2, "novice": 0.4}

        if self.level in expertise:
            drop_prob = expertise.get(self.level)

            user_slots = []
            for slot in filtered_slots:
                if np.random.random() > drop_prob:
                    user_slots.append(slot)
        else:
            raise ValueError("Unknown experience level: {}".format(self.level))

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
            goal_slots = goal.get('slots', list())

        inform_slot = find_slot_with_key(req_name, goal_slots)

        if inform_slot is None:
            return None

        user_act = build_user_act(UserAct.INFORM, None, [inform_slot])
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
            goal_slots = goal.get('slots', list())

        target_slot = find_slot_with_key(confirm_name, goal_slots)

        # What are you confirming?
        if target_slot is None:
            return None

        # Check values
        target_value = target_slot['value']
        if confirm_name in ["object_mask_str", "gesture_click"]:
            # Calculate dice_score
            mask_str = confirm_value
            goal_mask_str_slot = find_slot_with_key(confirm_name, goal_slots)
            goal_mask_str = goal_mask_str_slot['value']
            dice_score = self.compute_dice(mask_str, goal_mask_str)
            same = dice_score >= self.dice_threshold
        else:
            same = (confirm_value == target_value)

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

    def check_system_execution(self, execute_intent, execute_slots):
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

            if slot_name in ["object",
                             "gesture_click"]:  # Skip this slot in object_goal
                continue

            slot = find_slot_with_key(slot_name, execute_slots)
            if slot is None or not slot.get('value'):
                return False

            slot_value = slot['value']

            if slot_name == "object_mask_str":
                dice_score = self.compute_dice(slot_value, target_value)
                if dice_score < self.dice_threshold:
                    logger.info("dice score {} < dice threshold {}".format(
                        dice_score, self.dice_threshold))
                    return False
            else:
                if slot['value'] != target_value:
                    return False

        return True

    ###################
    #   Mask Related  #
    ###################
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
        assert mask.shape == goal_mask.shape
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
    return getattr(sys.modules[__name__], string)
