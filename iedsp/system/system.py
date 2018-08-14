from ..core import SystemAct
from .state import StatePortal
from ..util import find_slot_with_key, build_slot_dict


def build_sys_act(dialogue_act, intent=None, slots=None):
    sys_act = {}
    sys_act['dialogue_act'] = build_slot_dict("dialogue_act", dialogue_act)
    if intent is not None:
        sys_act['intent'] = build_slot_dict("intent", intent)
    if slots is not None:
        sys_act['slots'] = slots
    return sys_act


class System(object):
    """
    Mainly handles the interactions with the environment

    Attributes:
        state
        framestack
    """

    def __init__(self, global_config):
        self.state = StatePortal(global_config)

    def reset(self):
        """ 
        Resets for a new dialogue session
        """
        self.observation = {}
        self.state.clear()
        self.turn_id = 0

    def observe(self, observation):
        """
        If has only utterance, send to tracker for dialogue_act & slots 
        Args:
            observation (dict): observation given by user passed through channel
        """
        self.observation.update(observation)
        if 'user_acts' in observation:
            self.turn_id += 1

    def state_update(self):
        """
        Update dialogue state with user_acts
        """
        # Update actions from user
        photoshop_acts = self.observation.get('photoshop_acts', list())
        user_acts = self.observation.get('user_acts', list())
        observed_acts = photoshop_acts + user_acts

        for act in observed_acts:
            # Usually have only one user_act
            args = {
                'dialogue_act': act.get('dialogue_act', None),
                'intent': act.get('intent', None),
                'slots': act.get('slots', None),
                'turn_id': self.turn_id
            }
            self.state.update(**args)

    def policy(self):
        """
        A simple rule-based policy conditioned conditioned on the state
        Returns:
            system_acts (list): list of sys_acts
        """
        system_acts = []

        sysintent = self.state.pull()

        if len(sysintent.confirm_slots):
            da = SystemAct.CONFIRM
            intent = da
            slots = sysintent.confirm_slots[:1]  # confirm 1 at a time
        elif len(sysintent.request_slots):
            da = SystemAct.REQUEST
            intent = da
            slots = sysintent.request_slots[:1]  # request 1 at a time
        else:
            # Execute the intent
            da = SystemAct.EXECUTE
            intent = self.state.get_slot('intent').get_max_value()
            slots = sysintent.execute_slots

            # Stack the intent to history and clear intent slot
            self.state.stack_intent(intent)
            self.state.clear_slot('intent')

            # Clear graph if Undo or Redo
            if intent == "undo":
                self.state.clear_graph()

        # Request, Confirm, Label, Execute
        system_acts += [build_sys_act(da, intent, slots)]

        # Special cases
        if intent == "open":
            system_acts += [build_sys_act(SystemAct.GREETING)]
        if intent == "close":
            system_acts = [build_sys_act(SystemAct.BYE)]
        elif da == SystemAct.EXECUTE:
            system_acts += [build_sys_act(SystemAct.ASK)]
        return system_acts

    def act(self):
        """ 
        Perform action according to observation
        Returns:
            system_act (dict)
        """
        # Update state with observation
        self.state_update()

        # Policy
        system_acts = self.policy()

        # Update turn_id
        self.turn_id += 1

        # Build Return object
        system_act = {}
        system_act['system_acts'] = system_acts
        system_act['system_utterance'] = self.template_nlg(system_acts)
        system_act['episode_done'] = self.observation.get(
            'episode_done', False)

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
            elif sys_dialogue_act == SystemAct.EXECUTE:
                execute_slots = sys_act['slots']
                
                slot_list = [slot['slot'] + "=" + str(slot['value'])
                             for slot in execute_slots]
                utt = "Execute: " + ', '.join(slot_list) + "."
            elif sys_dialogue_act == SystemAct.BYE:
                utt = "Goodbye! See you next time!"
            else:
                utt = ""
            utt_list.append(utt)

        full_utterance = ' '.join(utt_list)
        return full_utterance
