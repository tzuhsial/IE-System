
from ..ontology import OntologyPortal
from .visionengine import VisionEnginePortal
from ..util import find_slot_with_key


def SystemPortal(config):

    visionengine = VisionEnginePortal(config['VISIONENGINE'])
    ontology = OntologyPortal(config["ONTOLOGY"])
    ontology.load_visionengine(visionengine)


    pass


class System(object):
    """
    Attributes:
        ontology (object)
        observation (dict)
    """

    def __init__(self, config):

        self.state = None
        # Storing stuff
        self.observation = {}

    def reset(self):
        """ 
        Resets for a new dialogue session
        """
        self.observation = {}
        self.state.clear()
        self.turn_id = 0

    def observe(self, observation):
        """
        Args:
            observation (dict): observation given by user passed through channel
        """
        self.observation.update(observation)
        self.turn_id += 1

    def state_update(self):
        """
        Update dialogue state with user_acts
        """
        # State Update
        user_acts = self.observation.get('user_acts')
        for user_act in user_acts:
            # Usually have only one user_act
            user_dialogue_act = user_act['dialogue_act']
            user_intent = user_act['intent']
            user_slots = user_act.get('slots')
            self.state.update(
                user_dialogue_act, user_intent, user_slots, self.turn_id)

    def policy(self):
        """
        A rule-based policy conditioned on dialogueState
        Returns:
            system_acts (list): list of sys_acts
        """
        # Here we implement a rule-based policy
        pass

    def act(self):
        """ 
        Perform action according to observation
        Returns:
            system_act (dict)
        """

        #b64_img_str = self.observation.get('b64_img_str', None)

        # Update state
        self.state_update()

        # Policy
        system_acts = self.policy()

        # Build Return object
        system_act = {}
        system_act['system_acts'] = system_acts
        system_act['system_utterance'] = self.template_nlg(system_acts)
        system_act['episode_done'] = self.observation['episode_done']
        system_act['speaker'] = self.SPEAKER
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
            sys_dialogue_act = sys_act['dialogue_act']['value']
            if sys_dialogue_act == SystemAct.GREETING:
                utt = "Hello! My name is PS. I am here to help you edit your image!"
            elif sys_dialogue_act == SystemAct.ASK:
                utt = "What would you like to do?"
            elif sys_dialogue_act == SystemAct.OOD_NL:
                utt = "Sorry, Photoshop currently does not support this function."
            elif sys_dialogue_act == SystemAct.OOD_CV:
                utt = "Sorry, Photoshop's CV engine cannot find what you are looking for."
            elif sys_dialogue_act == SystemAct.REPEAT:
                utt = "Can you please repeat again?"
            elif sys_dialogue_act == SystemAct.REQUEST:
                request_slots = [s['slot'] for s in sys_act['slots']]
                utt = "What " + ', '.join(request_slots) + " do you want?"
            elif sys_dialogue_act == SystemAct.CONFIRM:
                confirm_slots = sys_act['slots']
                utt = "Let me confirm. "
                confirm_list = []
                for slot_dict in confirm_slots:
                    sv = slot_dict['slot'] + " is " + str(slot_dict['value'])
                    confirm_list.append(sv)
                utt += ','.join(confirm_list) + "?"
            elif sys_dialogue_act == SystemAct.REQUEST_LABEL:
                utt = "I can not identify the object. Can you label it for me?"
            elif sys_dialogue_act == SystemAct.EXECUTE:
                execute_slots = sys_act['slots']

                slot_list = [slot['slot'] + "=" + str(slot['value'])
                             for slot in execute_slots]
                utt = "Execute: " + ', '.join(slot_list)
            elif sys_dialogue_act == SystemAct.BYE:
                utt = "Goodbye! See you next time!"
            else:
                utt = ""
            utt_list.append(utt)

        full_utterance = ' '.join(utt_list)
        return full_utterance
