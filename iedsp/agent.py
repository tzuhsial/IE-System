"""
    Finite State Machine Agent version 2
"""

from transitions import Machine

from .cvengine import CVEngineAPI


def sys_act_builder(dialogue_act, slots=None):
    sys_act = {}
    sys_act['dialogue_act'] = dialogue_act
    if slots is not None:
        sys_act['slots'] = slots
    return sys_act


class RuleBasedDialogueManager(object):
    """
        Define finite state transitions here
    """
    states = [
        'start_session', 'ask_ier', 'confirm', 'request', 'query_cv_engine', 'execute_api', 'ask_user_label',
        'end_session'
    ]

    def __init__(self):
        """ Defines finite state machine transitions
            Here we use FSM to make sure all dialog flow is controlled
        """
        self.machine = Machine(
            model=self, states=self.states, initial='start_session')

        self.machine.add_transition(
            trigger='greeting', source='start_session', dest='ask_ier')
        self.machine.add_transition(
            trigger='outOfDomain', source='ask_ier', dest=None)
        self.machine.add_transition(
            trigger='zeroNLConf', source='ask_ier', dest=None)
        self.machine.add_transition(trigger='lowNLConf', source=[
                                    'ask_ier', 'request', 'confirm'], dest='confirm')
        self.machine.add_transition(trigger='highNLConf_missing', source=[
                                    'ask_ier', 'confirm'], dest='request')
        self.machine.add_transition(trigger='highNLConf_noMissing', source=[
                                    'ask_ier', 'confirm', 'request'], dest='query_cv_engine')

        self.machine.add_transition(
            trigger='highCVConf', source='query_cv_engine', dest='execute_api')
        self.machine.add_transition(
            trigger='lowCVConf', source='query_cv_engine', dest='ask_user_label')
        self.machine.add_transition(
            trigger='hasLabel', source='ask_user_label', dest='execute_api')

        self.machine.add_transition(
            trigger='noLabel', source='ask_user_label', dest='ask_ier')

        self.machine.add_transition(
            trigger='result', source='execute_api', dest='ask_ier')
        self.machine.add_transition(
            trigger='bye', source='ask_ier', dest='end_session')

    def on_enter_ask_ier(self):
        self.clear_ier()

    @property
    def name(self):
        return self.__class__.__name__

    def reset(self):
        """ Resets session
        """
        self.clear_ier()
        self.to_start_session()  # Check on_enter_ask_ier for more ops

    def print_status(self):
        print('state', self.state)
        print('request_slots', self.request_slots)
        print('confirm_slots', self.confirm_slots)
        print('query_slots', self.query_slots)

    def clear_ier(self):
        """ Clear ier information
        """
        self.request_slots = []
        self.confirm_slots = []
        self.query_slots = []

        self.mask_slots = []

    def observe(self, observation):
        """ Updates observation and clears actions
        """
        self.observation = observation

    def act(self):
        """ Handle actions according to state
        """
        if self.state == "start_session":
            return self.act_start_session()
        elif self.state == "ask_ier":
            return self.act_ask_ier()
        elif self.state == "confirm":
            return self.act_confirm()
        elif self.state == "request":
            return self.act_request()
        elif self.state == "ask_user_label":
            return self.act_ask_user_label()
        else:
            raise ValueError("Unknown state {}!".format(self.state))

    ########################################
    #   Different act according to states  #
    ########################################
    def observe_apply_return(method):
        """Decorator
            Observes user act, applies method, then builds return object
        """
        def wrapper(agent):
            user_acts = agent.observation.get('user_acts')

            system_acts, episode_done = method(agent, user_acts)

            agent_act = {}
            agent_act['system_acts'] = system_acts
            agent_act['system_utterance'] = agent.template_nlg(system_acts)
            agent_act['episode_done'] = episode_done
            return agent_act
        return wrapper

    @observe_apply_return
    def act_start_session(self, user_acts):
        assert len(user_acts) == 1
        assert user_acts[0]['dialogue_act'] == "open"
        self.greeting()
        sys_act = sys_act_builder('greeting')
        sys_act2 = sys_act_builder('ask')
        return [sys_act, sys_act2], False

    @observe_apply_return
    def act_ask_ier(self, user_acts):
        """
            user_act inform should only have one act
            Possible sys_acts
            - confirm
            - request
            - query
            - bye
        """
        assert len(user_acts) == 1
        user_act = user_acts[0]

        system_acts = []
        episode_done = False

        dialogue_act = user_act['dialogue_act']
        if dialogue_act == "inform":
            self.update_slots(user_act['slots'])
            sys_act = self.nl_transition()
        elif dialogue_act == "bye":
            self.bye()
            sys_act = sys_act_builder('bye')
            episode_done = True
        system_acts.append(sys_act)
        return system_acts, episode_done

    @observe_apply_return
    def act_confirm(self, user_acts):
        """ actions in confirm state
            Possible sys_acts:
            - confirm
            - request
            - query
        """
        system_acts = []
        for user_act in user_acts:
            user_dialogue_act = user_act['dialogue_act']
            if user_dialogue_act == "affirm":
                # Responding to previous choice
                self.query_slots += self.confirm_slots[:1]
                self.confirm_slots.pop(0)
            elif user_dialogue_act == "negate":
                # Responding to previous choice
                self.request_slots += [d['slot']
                                       for d in self.confirm_slots[:1]]
                self.confirm_slots.pop(0)
            elif user_dialogue_act == "inform":
                self.update_slots(user_act['slots'])

        # Build sys_act
        sys_act = self.nl_transition()
        system_acts.append(sys_act)
        return system_acts, False

    @observe_apply_return
    def act_request(self, user_acts):
        """ State: request
            Possible sys_acts:
            - confirm
            - request
            - execute
        """
        system_acts = []
        assert len(user_acts) == 1
        user_act = user_acts[0]
        assert user_act['dialogue_act'] == 'inform'

        self.update_slots(user_act['slots'])

        # Build sys_act
        sys_act = self.nl_transition()
        system_acts.append(sys_act)

        return system_acts, False

    def nl_transition(self):
        """ Performs state transitions according to slot status
            Returns a sys_act object
        """
        # Build sys_act
        if self.hasConfirmSlots():
            self.lowNLConf()
            sys_act = sys_act_builder('confirm', self.confirm_slots[:1])
        elif self.hasRequestSlots():
            self.highNLConf_missing()
            sys_act = sys_act_builder('request', self.request_slots)
        elif len(self.query_slots) == 0:  # Nothing learned
            self.zeroNLConf()
            sys_act = sys_act_builder('repeat')
        else:
            self.highNLConf_noMissing()

            has_select = self.act_query()

            if has_select and len(self.mask_slots) != 1:
                self.lowCVConf()
                sys_act = sys_act_builder(
                    'request_label', slots=self.mask_slots)
            else:
                self.highCVConf()
                sys_act = sys_act_builder(
                    'execute', self.query_slots + self.mask_slots)
                self.result()
        return sys_act

    @staticmethod
    def find_slot_with_key(key, slots):
        # First check is edit or control
        for idx, slot_dict in enumerate(slots):
            if slot_dict['slot'] == key:
                return idx, slot_dict
        return -1, None

    def update_slots(self, inform_slots):
        """ Updates request, confirm, query, select slots
        """
        action_idx, action_slot = self.find_slot_with_key(
            'action_type', inform_slots)
        if action_idx < 0:  # Not found, look for existing
            _, action_slot = self.find_slot_with_key(
                'action_type', self.query_slots)

        action_type = action_slot['value']
        action_conf = action_slot['conf']
        if action_conf < 0.5:
            return

        if action_type == "adjust":
            query_slot_names = ['action_type',
                                'attribute', 'adjustValue', 'select']
        elif action_type in ["select"]:
            query_slot_names = ['select']
        elif action_type in ["redo", "undo"]:
            query_slot_names = ['action_type']
        else:
            raise NotImplementedError(
                "Unknown action_type: {}".format(action_type))

        for query_name in query_slot_names:
            inform_idx, slot = self.find_slot_with_key(
                query_name, inform_slots)
            if inform_idx < 0:
                if query_name not in self.request_slots and \
                        not any(query_name == d['slot'] for d in self.confirm_slots) and \
                        not any(query_name == d['slot'] for d in self.query_slots):
                    self.request_slots.append(query_name)
            else:  # Found in inform
                if any(query_name == d['slot'] for d in self.confirm_slots):
                    confirm_idx, _ = self.find_slot_with_key(
                        query_name, self.confirm_slots)
                    self.confirm_slots.pop(confirm_idx)
                if query_name in self.request_slots:
                    self.request_slots.remove(query_name)

                if slot['conf'] >= 0.8:
                    self.query_slots.append(slot)
                elif slot['conf'] >= 0.5:
                    self.confirm_slots.append(slot)
                else:  # Do nothing
                    pass

    def hasConfirmSlots(self):
        return len(self.confirm_slots) != 0

    def hasRequestSlots(self):
        return len(self.request_slots) != 0

    ###########################
    #       CV Related        #
    ###########################
    def act_query(self):
        """queries cv engine to obtain masks
           returns whether select is present
        """

        select_idx, select_slot = self.find_slot_with_key(
            'select', self.query_slots)
        if select_idx < 0:
            return False

        b64_img_str = self.observation.get('b64_img_str')

        noun = select_slot['value']

        masks = CVEngineAPI.fake_select(noun, b64_img_str)

        self.mask_slots.clear()
        for idx, mask_str in enumerate(masks):
            mask_name = 'mask' + str(idx)
            mask_slot = {'slot': mask_name, 'value': mask_str}
            self.mask_slots.append(mask_slot)

        return True

    @observe_apply_return
    def act_ask_user_label(self, user_acts):
        system_acts = []
        for user_act in user_acts:
            user_dialogue_act = user_act['dialogue_act']
            if user_dialogue_act == "negate":
                self.noLabel()
                self.clear_ier()
                sys_act = sys_act_builder('clear_edit')
            elif user_dialogue_act == "tool_select":
                self.hasLabel()
                select_slots = user_act['slots']

                selected_mask_slots = []
                for slot in select_slots:
                    idx, mask_slot = self.find_slot_with_key(
                        slot['value'], self.mask_slots)
                    if idx >= 0:
                        selected_mask_slots.append(mask_slot)

                self.mask_slots = selected_mask_slots

                #query_mask_slots = [ d['value'] for d in self.mask_slots ]

                sys_act = sys_act_builder(
                    'execute', self.query_slots + self.mask_slots)
                self.result()

            system_acts.append(sys_act)
        return system_acts, False

    ###########################
    #   Simple Template NLG   #
    ###########################

    def template_nlg(self, system_acts):
        """Given system_acts, return an utterance string
        """
        utt_list = []
        for sys_act in system_acts:
            if sys_act['dialogue_act'] == 'greeting':
                utt = "Hello! My name is PS. I am here to help you edit your image!"
            elif sys_act['dialogue_act'] == 'ask':
                utt = "What would you like to do?"
            elif sys_act['dialogue_act'] == 'ood':
                utt = "Sorry, Photoshop currently does not support this function."
            elif sys_act['dialogue_act'] == 'repeat':
                utt = "Can you please repeat again?"
            elif sys_act['dialogue_act'] == 'request':
                request_slots = sys_act['slots']
                utt = "What " + ', '.join(request_slots) + " do you want?"
            elif sys_act['dialogue_act'] == 'confirm':
                confirm_slots = sys_act['slots']
                utt = "Let me confirm. "

                confirm_list = []
                for slot_dict in confirm_slots:
                    sv = slot_dict['slot'] + " is " + slot_dict['value']
                    confirm_list.append(sv)

                utt += ','.join(confirm_list) + "?"

            elif sys_act['dialogue_act'] == 'request_label':
                mask_slots = sys_act['slots']
                if len(mask_slots) == 0:
                    utt = "I can not identify the object. Can you label it for me?"
                else:
                    utt = "I can not identify the object. Can you select it for me?"
            elif sys_act['dialogue_act'] == 'execute':
                execute_slots = sys_act['slots']

                slot_list = [slot['slot'] + "=" + slot['value']
                             for slot in execute_slots]
                utt = "Execute: " + ', '.join(slot_list)
            elif sys_act['dialogue_act'] == 'clear_edit':
                utt = "Edit forfeited. Status cleared."
            elif sys_act['dialogue_act'] == 'bye':
                utt = "Goodbye! See you next time!"
            else:
                utt = ""
            utt_list.append(utt)

        full_utterance = ' '.join(utt_list)
        return full_utterance
