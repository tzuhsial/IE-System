from ..core import SystemAct
from .state import StatePortal
from ..visionengine import VisionEnginePortal
from ..util import find_slot_with_key, build_slot_dict, slots_to_args


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
        self.visionengine = VisionEnginePortal(global_config["VISIONENGINE"])

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
        A simple rule-based policy conditioned conditioned on state
        Returns:
            system_acts (list): list of sys_acts
        """
        system_acts = []

        # Pull the current intent slots =>
        # Rules are defined in the nodes
        sysintent = self.state.pull()

        if len(sysintent.confirm_slots):
            sys_act = self.act_confirm(sysintent.confirm_slots)
        elif len(sysintent.request_slots):
            sys_act = self.act_request(sysintent.request_slots)
        elif len(sysintent.query_slots):
            sys_act = self.act_query(sysintent.query_slots)
        else:
            sys_act = self.act_execute(sysintent.execute_slots)

        # Request, Confirm, Label, Execute

        system_acts += [sys_act]

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

    def act_confirm(self, confirm_slots):
        da = SystemAct.CONFIRM
        intent = da
        slots = confirm_slots[:1]  # confirm 1 at a time
        sys_act = build_sys_act(da, intent, slots)
        return sys_act

    def act_request(self, request_slots):
        da = SystemAct.REQUEST
        intent = da
        slots = request_slots[:1]  # request 1 at a time
        sys_act = build_sys_act(da, intent, slots)
        return sys_act

    def act_query(self, query_slots):

        da = SystemAct.QUERY
        intent = da
        slots = query_slots
        sys_act = build_sys_act(da, intent, slots)

        #########################
        #  Query Vision Engine  #
        #########################
        args = slots_to_args(query_slots)
        b64_img_str = self.state.get_slot('b64_img_str').get_max_value()
        args['b64_img_str'] = b64_img_str

        mask_strs = self.visionengine.select_object(**args)
        object_mask_str_node = self.state.get_slot('object_mask_str')

        # Clear object_mask_results & object_mask_id
        object_mask_str_node.value_conf_map.clear()
        self.state.get_slot('object_mask_id').clear()
        if len(mask_strs) > 0:
            object_mask_str_node.value_conf_map = \
                {mask_str: 0.5 for mask_str in mask_strs}

        print("query results: ", len(mask_strs))
        object_mask_str_node.last_update_turn_id += 1

        return sys_act

    def act_execute(self, execute_slots):
        # Build execute sys_act
        da = SystemAct.EXECUTE
        intent = self.state.get_slot('intent').get_max_value()
        slots = execute_slots.copy()  # Make a copy, since clearing will remove this copy

        sys_act = build_sys_act(da, intent, slots)

        # Stack the intent to history and clear intent slot
        self.state.stack_intent(intent)
        self.state.clear_graph()

        return sys_act

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
                    if slot_dict['slot'] == "object_mask_str":
                        sv = slot_dict['slot'] + " is " + \
                            str(slot_dict['value'][:5])
                    else:
                        sv = slot_dict['slot'] + " is " + \
                            str(slot_dict['value'])
                    confirm_list.append(sv)
                utt += ','.join(confirm_list) + "?"
            elif sys_dialogue_act == SystemAct.QUERY:
                query_slots = sys_act['slots']
                slot_list = [slot['slot'] + "=" + str(slot['value'])
                             for slot in query_slots]
                utt = "Query: " + ', '.join(slot_list) + "."
            elif sys_dialogue_act == SystemAct.EXECUTE:
                execute_slots = sys_act['slots']

                slot_list = []
                for slot in execute_slots:
                    if slot["slot"] == "object_mask_str":
                        sv = slot['slot'] + "=" + str(slot['value'][:5])
                    else:
                        sv = slot['slot'] + "=" + str(slot['value'])
                    slot_list.append(sv)
                utt = "Execute: " + ', '.join(slot_list) + "."
            elif sys_dialogue_act == SystemAct.BYE:
                utt = "Goodbye! See you next time!"
            else:
                utt = ""
            utt_list.append(utt)

        full_utterance = ' '.join(utt_list)
        return full_utterance
