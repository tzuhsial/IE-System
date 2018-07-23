import copy
import random

import numpy as np

from ..common import build_act
from ..dialogueacts import UserAct, SystemAct
from ..ontology import ImageEditOntology
from ..util import find_slot_with_key, b64_to_img


def user_act_builder(dialogue_act, intent, slots=None):
    """Build user_act list with dialogue_act and slots
    """
    user_act = {}
    user_act['dialogue_act'] = dialogue_act
    user_act['intent'] = intent
    if slots is not None:
        user_act['slots'] = slots
    return user_act


class AgendaBasedUserSimulator(object):
    """
    An implementation of an agenda based user simulator
    for image editing requests.
    """

    def __init__(self, config):
        """
        Initialize user simulator with configuration
        """
        self.dice_threshold = float(config['DICE_THRESHOLD'])
        self.negate_inform_prob = float(config['NEGATE_INFORM_PROB'])

    def load_agenda(self, agenda):
        """
        Args:
            agenda (list): list of goals
        """
        self.agenda = agenda

    def print_agenda(self):
        """
        Displays current agenda in human readable format
        """
        for agenda in self.agenda:
            print('intent', agenda['intent'])

    def curr_agenda(self):
        """
        Returns:
            agenda (dict)
        """
        if self.agenda is None:
            return None
        return self.agenda[0]

    def reset(self):
        """
        Reset user simulator
        """
        self.agenda = None
        self.observation = {}

    def observe(self, observation):
        """
        Updates observation
        """
        self.observation.update(observation)

    def act(self):
        """
        Agenda based action with configurations
        """
        assert self.agenda is None or len(self.agenda) > 0
        user_acts = []

        if len(self.observation) == 0:
            # Observation is empty => Open an image
            user_act = self.act_open()
            user_acts.append(user_act)
        else:
            system_acts = self.observation.get('system_acts', [])
            for sys_act in system_acts:
                if sys_act['dialogue_act'] == SystemAct.REQUEST:

                    import pdb
                    pdb.set_trace()
                elif sys_act['dialogue_act'] == SystemAct.CONFIRM:
                    assert len(
                        sys_act['slots']) == 1, "[User] System should only confirm one slot at a time!"
                    confirm_slot = sys_act['slots'][0]
                    confirm_slot_name = confirm_slot['slot']
                    confirm_slot_value = confirm_slot['value']

                    _, target_slot = find_slot_with_key(
                        confirm_slot_name, self.agenda[0]['slots'])

                    target_slot_name = target_slot['slot']
                    if target_slot_name == "object":
                        target_slot_value = target_slot['value']['name']
                    else:
                        target_slot_value = target_slot['value']

                    if confirm_slot_value == target_slot_value:
                        user_act = self.act_affirm()
                    else:
                        user_act = self.act_negate()
                    user_acts.append(user_act)
                elif sys_act['dialogue_act'] == SystemAct.REQUEST_LABEL:
                    print("SystemAct.REQUEST_LABEL")

                    import pdb
                    pdb.set_trace()
                elif sys_act['dialogue_act'] == SystemAct.EXECUTE:
                    # Previous agenda has been executed
                    self.agenda.pop(0)
                    if self.agenda[0]['intent'] == "close":
                        user_act = self.act_close()
                    else:
                        user_act = self.act_inform_agenda()
                    user_acts.append(user_act)

        # Build return object
        user_act = {}
        user_act['user_acts'] = user_acts
        user_act['user_utterance'] = self.template_nlg(user_acts)
        user_act['episode_done'] = self.agenda[0]['intent'] == "close"
        return user_act

    def act_open(self):
        """
        Returns:
            user_act : open image user act
        """
        assert len(self.observation) == 0
        currAgenda = self.agenda[0]
        assert currAgenda['intent'] == ImageEditOntology.OPEN

        user_act = copy.deepcopy(currAgenda)
        user_act['dialogue_act'] = UserAct.OPEN
        return user_act

    def act_close(self):
        """
        Returns:
            user_act: close image user act
        """
        assert len(self.agenda) == 1
        currAgenda = self.agenda[0]
        assert currAgenda['intent'] == ImageEditOntology.CLOSE

        user_act = copy.deepcopy(currAgenda)
        user_act['dialogue_act'] = UserAct.CLOSE
        return user_act

    def act_inform_agenda(self):
        """
        Returns:
            user_act (dict): user act object
        """
        agenda = self.agenda[0]

        user_act = copy.deepcopy(agenda)
        user_act['dialogue_act'] = UserAct.INFORM
        user_intent = user_act['intent']
        if user_intent in ["adjust", "select_object"]:
            # Set object value from { "mask_str": mask_str, "name": name } to name
            _, object_slot = find_slot_with_key('object', user_act['slots'])
            object_slot['value'] = object_slot['value']['name']

            # TODO: Number of slots filtered decided on probability
        elif user_intent in ["undo", "redo"]:
            # We don't have any slot values
            pass
        return user_act

    def act_affirm(self):
        """
        Returns:
            user_act (dict)
        """
        user_act = {}
        user_act['dialogue_act'] = UserAct.AFFIRM
        return user_act

    def act_negate(self):
        """
        Returns:
            user_act (dict)
        """
        user_act = {}
        user_act['dialogue_act'] = UserAct.NEGATE
        return user_act

    def act_label(self):
        """
        Returns:
            user_act (dict)
        """
        user_act = {}
        user_act['dialogue_act'] = UserAct.LABEL
        user_act['intent'] = "label"

        import pdb
        pdb.set_trace()
        return user_act

    def compute_dice(self, mask_str, target_mask_str=None):
        """
        Computes the dice metric between the mask_str and the target_mask_str
        If target_mask_str is not provided, will extract it from current agenda
        Args:
            mask_str (str): b64 image string
            target_mask_str (str): b64 image string
        Returns:
            dice (float): dice metric
        """
        if target_mask_str is None:
            _, object_slot = find_slot_with_key("object", self.agenda[0])
            target_mask_str = object_slot['value']['mask_str']

        mask = b64_to_img(mask_str)
        target_mask = b64_to_img(target_mask_str)
        assert mask.shape == target_mask.shape, "Weird shape"

        dice = 2 * ( mask & target_mask ).sum() / ( mask | target_mask).sum()

        return dice >= self.dice_threshold

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
            user_dialogue_act = user_act['dialogue_act']
            if user_dialogue_act == UserAct.OPEN:
                utt = "(Open an image)"
            elif user_dialogue_act == UserAct.INFORM:
                slots = user_act['slots']
                slot_list = []
                for slot in slots:
                    slot_list.append(
                        slot['slot'] + " to be " + str(slot['value']))
                utt = "I want " + ', '.join(slot_list) + "."
            elif user_dialogue_act == UserAct.AFFIRM:
                utt = "Yes."
            elif user_dialogue_act == UserAct.NEGATE:
                utt = "No."
            elif user_dialogue_act == UserAct.TOOL_SELECT:
                slots = user_act['slots']
                slot_list = []
                for slot in slots:
                    slot_list.append(slot['value'])
                utt = "I want to select " + ", ".join(slot_list)
            elif user_dialogue_act == UserAct.CLOSE:
                utt = "Bye."
            elif user_dialogue_act == "select_object_mask_id":
                slot = user_act['slots'][0]
                mask_id = slot['value']
                utt = "I want to select object {}.".format(mask_id)
            else:
                raise ValueError(
                    "Unknown user_dialogue_act: {}".format(user_dialogue_act))
            utt_list.append(utt)

        utterance = ' '.join(utt_list)
        return utterance


class DemoUser(object):
    """
        This user acts as the interface to the server
    """

    def __init__(self):
        raise NotImplementedError
