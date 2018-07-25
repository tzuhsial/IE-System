import copy
import random

import numpy as np

from ..core import Agent, UserAct, SystemAct, Hermes
from ..ontology import ImageEditOntology
from ..util import find_slot_with_key, b64_to_img


class AgendaBasedUserSimulator(object):
    """
    An implementation of an agenda based user simulator
    for image editing requests.
    """

    SPEAKER = Agent.USER

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
        Perform actions according to the current agenda
        """
        assert self.agenda is None or len(self.agenda) > 0

        user_acts = []
        # The default action for the user is to inform the agenda
        default_system_acts = [Hermes.build_act(SystemAct.ASK)]
        system_acts = self.observation.get('system_acts', default_system_acts)
        for sys_act in system_acts:
            # We can do a for loop here of the system acts
            # because there can only be one system act that
            # requires a response from the user
            sys_dialogue_act = sys_act['dialogue_act']['value']
            if sys_dialogue_act == SystemAct.ASK:
                user_act = self.act_inform_agenda()
                user_acts += [user_act]
            elif sys_dialogue_act == SystemAct.REQUEST:
                request_slots = sys_act['slots']
                user_act = self.act_inform_agenda(request_slots)
            elif sys_dialogue_act == SystemAct.CONFIRM:
                assert len(sys_act['slots']) == 1
                confirm_slot = sys_act['slots'][0]
                confirm_name = confirm_slot['slot']
                confirm_value = confirm_slot['value']

                # Confirm intent or normal slot values
                if confirm_name == ImageEditOntology.DIALOGUE_ACT:
                    slots = [self.agenda[0]['dialogue_act']]
                else:
                    slots = self.agenda[0]['slots']

                _, target_slot = find_slot_with_key(confirm_name, slots)

                # Special case: object
                target_name = target_slot['slot']
                if target_name == ImageEditOntology.Slots.OBJECT:
                    target_value = target_slot['value']['name']
                else:
                    target_value = target_slot['value']

                # Decide Affirm or Negate
                if confirm_value == target_value:
                    user_dialogue_act = UserAct.AFFIRM
                else:
                    user_dialogue_act = UserAct.NEGATE

                # Build user_act
                user_act = Hermes.build_act(user_dialogue_act)
                user_acts += [user_act]
                # TODO: Probability of additional inform
            elif sys_dialogue_act == SystemAct.REQUEST_LABEL:
                print("SystemAct.REQUEST_LABEL")
                import pdb
                pdb.set_trace()

            elif sys_dialogue_act == SystemAct.EXECUTE:
                # Previous agenda has been executed
                self.agenda.pop(0)

        # Build return object
        user_act = {}
        user_act['user_acts'] = user_acts
        user_act['user_utterance'] = self.template_nlg(user_acts)
        user_act['episode_done'] = self.agenda[0]['dialogue_act']['value'] == UserAct.CLOSE
        user_act['speaker'] = self.SPEAKER
        return user_act

    def act_inform_agenda(self, request_slots=None):
        """
        Informs according to agenda, if request_slots are provided, 
        then inform only slots in request
        Returns:
            user_act (dict): user act object
        """
        agenda = self.agenda[0]
        user_act = copy.deepcopy(agenda)
        user_dialogue_act = user_act['dialogue_act']['value']
        user_act['speaker'] = Agent.USER
        if request_slots is not None:
            # Filter only the slots requested
            slots = []
            for request_slot in request_slots:
                _, inform_slot = find_slot_with_key(
                    request_slot['slot'], user_act['slots'])
                slots.append(inform_slot)
            user_act['slots'] = slots

        # Special case: object
        if user_dialogue_act == UserAct.ADJUST:
            # Set object value from { "mask_str": mask_str, "name": name } to name
            _, object_slot = find_slot_with_key('object', user_act['slots'])
            object_slot['value'] = object_slot['value']['name']
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

        dice = 2 * (mask & target_mask).sum() / (mask | target_mask).sum()

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
            user_dialogue_act = user_act['dialogue_act']['value']
            if user_dialogue_act == UserAct.OPEN:
                utt = "(Open an image)"
            elif user_dialogue_act == UserAct.LOAD:
                utt = "(Loads an image)"
            elif user_dialogue_act == UserAct.ADJUST:
                slots = user_act['slots']
                slot_list = ["dialogue_act to be " + user_dialogue_act]
                for slot in slots:
                    slot_list.append(
                        slot['slot'] + " to be " + str(slot['value']))
                utt = "I want " + ', '.join(slot_list) + "."
            elif user_dialogue_act == UserAct.AFFIRM:
                utt = "Yes."
            elif user_dialogue_act == UserAct.NEGATE:
                utt = "No."
            elif user_dialogue_act == UserAct.CLOSE:
                utt = "(Closes the image)"
            elif user_dialogue_act == UserAct.UNDO:
                utt = "I want to undo my previous edit."
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
