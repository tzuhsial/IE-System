import copy
import logging
import random
import sys

import numpy as np

from ..core import UserAct, SystemAct
from ..util import find_slot_with_key, b64_to_img, build_slot_dict

logger = logging.getLogger(__name__)


def UserPortal(user_config):
    user_type = user_config["USER"]
    dice_threshold = float(user_config["DICE_THRESHOLD"])
    select_prob = float(user_config["SELECT_PROB"])
    return builder(user_type)(select_prob, dice_threshold)


class AgendaBasedUserSimulator(object):
    """
    An implementation of an agenda based user simulator
    for image editing requests.
    """

    def __init__(self, select_prob, dice_threshold):
        """
        Initialize user simulator with configuration
        """
        self.select_prob = select_prob
        self.dice_threshold = dice_threshold

    def load_agenda(self, agenda):
        """
        Args:
            agenda (list): list of goals
        """
        self.agenda = agenda

    def process_agenda(self, agenda):
        raise NotImplementedError

    def print_agenda(self):
        """
        Displays current agenda in human readable format
        """
        for agenda in self.agenda:
            print('intent', agenda['intent'])

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
            {'dialogue_act': build_slot_dict('dialogue_act', SystemAct.ASK)}]

        # We needs system_acts
        system_acts = self.observation.get('system_acts', default_system_acts)
        for sys_act in system_acts:
            # We can do a for loop here of the system acts
            # because there can only be one system act that
            # requires a response from the user
            sys_dialogue_act = sys_act['dialogue_act']['value']
            if sys_dialogue_act == SystemAct.ASK:
                user_act = self.act_inform()
                user_acts.append(user_act)

            elif sys_dialogue_act == SystemAct.REQUEST:
                request_slot = sys_act["slots"][0]
                raise NotImplementedError

            elif sys_dialogue_act == SystemAct.CONFIRM:
                raise NotImplementedError

            elif sys_dialogue_act == SystemAct.REQUEST_LABEL:
                label_slots = sys_act["slots"]
                user_act = self.act_label(label_slots)
                user_acts.append(user_act)

            elif sys_dialogue_act == SystemAct.EXECUTE:
                exec_intent = sys_act["intent"]
                exec_slots = sys_act["slots"]
                success = self.check_execution_result(exec_intent, exec_slots)

                print("Execution result", success)
                if not success:
                    user_act = {}
                    user_act["dialogue_act"] = build_slot_dict(
                        'dialogue_act', 'inform')
                    user_act["intent"] = build_slot_dict("intent", "undo")
                    user_acts.append(user_act)
                else:
                    self.agenda.pop(0)

        # Check episode_done based on goal
        curr_goal = self.get_current_goal()
        episode_done = curr_goal["intent"]["value"] == "close"

        # Build return object
        user_act = {}
        user_act['user_acts'] = user_acts
        user_act['user_utterance'] = self.template_nlg(user_acts)
        user_act['episode_done'] = episode_done
        return user_act

    def act_inform(self, slots=None):
        """
        Informs the current goal, which is the top of the agenda
        Args:
            slots (list): slots that needs to be informed
        Returns:
            user_act (dict): user act object
        """
        # Get current goal
        goal = self.get_current_goal()

        # user_act
        user_act = copy.deepcopy(goal)
        user_act["dialogue_act"] = build_slot_dict(
            "dialogue_act", UserAct.INFORM)

        user_slots = user_act.get("slots", list())
        user_slots = filter(
            lambda slot: slot['slot'] != 'object_mask_str', user_slots)
        user_act["slots"] = list(user_slots)
        if slots is not None:
            user_act["slots"] = slots

        return user_act

    def act_label(self, label_slots):
        """
        Response to System.REQUEST_LABEL
        """
        goal = self.get_current_goal()
        goal_slots = goal.get("slots", list())
        goal_mask_str_slot = find_slot_with_key('object_mask_str', goal_slots)

        if goal_mask_str_slot is None:
            # User wants to edit the whole image
            # but the system asks the user to label...
            object_slot = find_slot_with_key('object', goal_slots)
            user_act = self.act_inform([object_slot])
            return user_act

        if len(label_slots) and all(slot.get('value') for slot in label_slots):
            # Compute dice score and select the one with the highest

            goal_mask_str = goal_mask_str_slot["value"]

            scores = []
            for idx, mask_str_slot in enumerate(label_slots):
                mask_str = mask_str_slot["value"]
                dice_score = self.compute_dice(mask_str, goal_mask_str)
                scores.append((idx, dice_score))

            # Sort scores with descending order
            sorted_scores = sorted(
                scores, key=lambda tup: tup[1], reverse=True)

            # Check score
            max_idx, max_score = sorted_scores[0]

            if max_score >= self.dice_threshold:
                object_mask_id_slot = build_slot_dict(
                    'object_mask_id', max_idx)
                user_act = self.act_inform([object_mask_id_slot])

                goal_object_mask_id_slot = find_slot_with_key(
                    'object_mask_id', goal_slots)

                if goal_object_mask_id_slot is None:
                    goal_slots.append(object_mask_id_slot)
                else:
                    goal_object_mask_id_slot["value"] = max_idx

            else:
                # Directly label it
                user_act = self.act_inform([goal_mask_str_slot])
        else:
            user_act = self.act_inform([goal_mask_str_slot])

        return user_act

    def check_execution_result(self, execute_intent, execute_slots):
        """
        Check if system execution result matches current goal
        Returns:
            success (bool): True if success else False
        """
        success = True
        goal = self.get_current_goal()

        # Check intents
        if execute_intent['value'] != goal['intent']['value']:
            success == False
        elif len(execute_slots) != len(goal['slots']):
            success == False
        else:
            for target_slot in goal["slots"]:
                slot_name = target_slot['slot']
                slot = find_slot_with_key(slot_name, execute_slots)

                if slot is None:
                    logger.debug("Missing slot: {}".format(slot_name))
                    success = False
                    break

                if slot_name == "mask_str":
                    # Compute the dice score between mask_str & target_mask_str
                    mask_str = slot['value']
                    target_mask_str = target_slot["value"]
                    dice_score = self.compute_dice(mask_str, target_mask_str)
                    if dice_score < self.dice_threshold:
                        success = False
                        break

                if slot["value"] != target_slot["value"]:
                    success = False
                    break

        return success

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
        dice = 2 * (mask & goal_mask).sum() / (mask | goal_mask).sum()
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
                intent_slot = user_act['intent']
                slots = [intent_slot] + user_act["slots"]
                slot_list = []
                for slot in slots:
                    slot_list.append(slot['slot'] + "=" + str(slot['value']))
                utt = "I want " + ', '.join(slot_list) + "."
            elif user_dialogue_act == UserAct.AFFIRM:
                utt = "Yes."
            elif user_dialogue_act == UserAct.NEGATE:
                utt = "No."
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
