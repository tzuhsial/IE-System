"""
    Finite State Machine Agent version 2
"""

from transitions import Machine

from ..common import build_act
from ..cvengine import CVEngineClient
from ..dialogueacts import SystemAct, UserAct
from .dialoguestate import DialogueState
from ..ontology import getOntologyByName
from ..util import find_slot_with_key


def sys_act_builder(dialogue_act, intent, slots=None):
    """Build sys_act list with dialogue_act and slots
    """
    sys_act = {}
    sys_act['dialogue_act'] = dialogue_act
    sys_act['intent'] = intent
    if slots is not None:
        sys_act['slots'] = slots
    return sys_act


class RuleBasedDialogueManager(object):
    """
    Rule based dialogue manager

    Attributes:
        confirm_threshold (float): the confidence score that decides whether to confirm
        cvengine (object): computer vision engine client
        ontology (object)
        dialogueState (object)
        observation (dict)
    """

    def __init__(self, config):
        super(RuleBasedDialogueManager, self).__init__()

        # Configurations
        self.confirm_threshold = float(config['CONFIRM_THRESHOLD'])
        self.cvengine = CVEngineClient(config['CVENGINE_URI'])
        self.ontology = getOntologyByName(config['ONTOLOGY'])

        # Construct dialogue state based on Ontology
        self.dialogueState = DialogueState(self.ontology)

        # Storing stuff
        self.observation = {}

    def reset(self):
        """ 
        Resets for a new dialogue session
        """
        self.observation = {}
        self.dialogueState.clear()
        self.turn_id = 0

    def observe(self, observation):
        """
        Args:
            observation (dict): observation given by user passed through channel
        """
        self.observation.update(observation)

    def act(self):
        """ 
        Perform action according to observation
        Returns:
            system_act (dict)
        """
        user_acts = self.observation.get('user_acts')
        episode_done = self.observation.get('episode_done', False)
        b64_img_str = self.observation.get('b64_img_str', None)

        # Here we implement a rule-based policy
        system_acts = []
        for user_act in user_acts:
            user_dialogue_act = user_act['dialogue_act']
            # Default: Out of domain
            user_intent = user_act.get('intent', self.ontology.OOD)
            # Default: Empty List
            user_slots = user_act.get('slots', list())

            self.dialogueState.update(user_intent, user_slots, self.turn_id)

            if user_dialogue_act == UserAct.OPEN:
                assert user_intent == self.ontology.OPEN
                # Issue Request
                sys_act = sys_act_builder(
                    SystemAct.EXECUTE, self.ontology.OPEN, user_slots)
                # Remove values from table
                self.dialogueState.clear(user_intent)
            elif user_dialogue_act in ["inform"]:
                # Get slot from current domain
                confirm_slots, request_slots, query_slots = \
                    self.dialogueState.getCurrentDomainTable().getCurrentSlots(self.confirm_threshold)

                if len(confirm_slots) > 0:
                    # Confirm one at a time
                    sys_act = sys_act_builder(
                        SystemAct.CONFIRM, 'confirm', confirm_slots[:1])
                elif len(request_slots) > 0:
                    # Request all
                    sys_act = sys_act_builder(
                        SystemAct.REQUEST, 'request', request_slots)
                else:
                    # Query CV Engine
                    _, object_slot = find_slot_with_key("object", query_slots)
                    noun = object_slot['value']
                    mask_strs = self.cvengine.select_object(noun, b64_img_str)

                    if len(mask_strs) == 0:
                        sys_act = sys_act_builder(
                            SystemAct.REQUEST_LABEL, 'request_label')
                    else:
                        # When CV engine return masks
                        sys_act = sys_act_builder(
                            SystemAct.REQUEST_LABEL, 'request_label')

            elif user_dialogue_act in ["affirm"]:
                confirm_slots, request_slots, query_slots = \
                    self.dialogueState.getCurrentDomainTable().getCurrentSlots(self.confirm_threshold)
            elif user_dialogue_act in ["negate"]:
                pass
            elif user_dialogue_act in ["close"]:
                sys_act = sys_act_builder(
                    SystemAct.BYE, self.ontology.CLOSE)  # No slots are needed
            else:
                raise ValueError(
                    "Unknown user_dialogue_act: {}".format(user_dialogue_act))

            system_acts.append(sys_act)

        system_act = {}
        system_act['system_acts'] = system_acts
        system_act['system_utterance'] = self.template_nlg(system_acts)
        system_act['episode_done'] = episode_done
        return system_act

    ###########################
    #   Simple Template NLG   #
    ###########################

    def template_nlg(self, system_acts):
        """
        Template based natural language generation
        """
        utt_list = []
        for sys_act in system_acts:
            if sys_act['dialogue_act'] == SystemAct.GREETING:
                utt = "Hello! My name is PS. I am here to help you edit your image!"
            elif sys_act['dialogue_act'] == SystemAct.ASK:
                utt = "What would you like to do?"
            elif sys_act['dialogue_act'] == SystemAct.OOD_NL:
                utt = "Sorry, Photoshop currently does not support this function."
            elif sys_act['dialogue_act'] == SystemAct.OOD_CV:
                utt = "Sorry, Photoshop's CV engine cannot find what you are looking for."
            elif sys_act['dialogue_act'] == SystemAct.REPEAT:
                utt = "Can you please repeat again?"
            elif sys_act['dialogue_act'] == SystemAct.REQUEST:
                request_slots = sys_act['slots']
                utt = "What " + ', '.join(request_slots) + " do you want?"
            elif sys_act['dialogue_act'] == SystemAct.CONFIRM:
                confirm_slots = sys_act['slots']
                utt = "Let me confirm. "

                confirm_list = []
                for slot_dict in confirm_slots:
                    sv = slot_dict['slot'] + " is " + str(slot_dict['value'])
                    confirm_list.append(sv)

                utt += ','.join(confirm_list) + "?"

            elif sys_act['dialogue_act'] == SystemAct.REQUEST_LABEL:
                utt = "I can not identify the object. Can you label it for me?"
            elif sys_act['dialogue_act'] == SystemAct.EXECUTE:
                execute_slots = sys_act['slots']

                slot_list = [slot['slot'] + "=" + slot['value']
                             for slot in execute_slots]
                utt = "Execute: " + ', '.join(slot_list)
            elif sys_act['dialogue_act'] == SystemAct.BYE:
                utt = "Goodbye! See you next time!"
            else:
                utt = ""
            utt_list.append(utt)

        full_utterance = ' '.join(utt_list)
        return full_utterance
