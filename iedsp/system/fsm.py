"""
    Finite State Machine Agent version 2
"""

from transitions import Machine

from ..cvengine import CVEngineAPI
from .dialoguestate import DialogueState
from ..ontology import getOntologyByName
from ..util import find_slot_with_key


def sys_act_builder(dialogue_act, slots=None):
    """Build sys_act list with dialogue_act and slots
    """
    sys_act = {}
    sys_act['dialogue_act'] = dialogue_act
    if slots is not None:
        sys_act['slots'] = slots
    return [sys_act]


class FiniteStateMachine(object):
    """
        Define the finite state transition for dialogue manager
    """
    states = [
        'start_session', 'ask_ier', 'confirm', 'request', 'ier_complete',
        'query_cv_engine', 'execute_api', 'ask_user_label', 'end_session'
    ]

    def __init__(self):
        """ Defines finite state machine transitions
            Here we use FSM to make sure all dialog flow is controlled
        """
        self.machine = Machine(
            model=self, states=self.states, initial='start_session')

        # NLU Part
        self.machine.add_transition(
            trigger='greeting', source='start_session', dest='ask_ier')
        self.machine.add_transition(
            trigger='outOfDomain', source='ask_ier', dest=None)
        self.machine.add_transition(
            trigger='zeroNLConf', source='ask_ier', dest=None)
        self.machine.add_transition(trigger='lowNLConf', source=[
                                    'ask_ier', 'request', 'confirm'], dest='confirm')
        self.machine.add_transition(trigger='highNLConf_missing', source=[
                                    'ask_ier', 'confirm', 'request'], dest='request')
        self.machine.add_transition(trigger='highNLConf_noMissing', source=[
                                    'ask_ier', 'confirm', 'request'], dest='ier_complete')

        # CV Part
        self.machine.add_transition(
            trigger='maskMissing', source='ier_complete', dest='query_cv_engine')
        self.machine.add_transition(
            trigger='highCVConf', source='query_cv_engine', dest='ier_complete')
        self.machine.add_transition(
            trigger='lowCVConf', source='query_cv_engine', dest='ask_user_label')
        self.machine.add_transition(
            trigger='hasLabel', source='ask_user_label', dest='ier_complete')
        self.machine.add_transition(
            trigger='noLabel', source='ask_user_label', dest='ask_ier')

        # Execute
        self.machine.add_transition(
            trigger='noMaskMissing', source="ier_complete", dest="execute_api")
        self.machine.add_transition(
            trigger='result', source='execute_api', dest='ask_ier')
        self.machine.add_transition(
            trigger='bye', source='ask_ier', dest='end_session')


class RuleBasedDialogueManager(FiniteStateMachine):
    def __init__(self, config):
        super(RuleBasedDialogueManager, self).__init__()

        # Configurations
        self.confirm_threshold = config['CONFIRM_THRESHOLD']
        self.cvengine = CVEngineAPI(config['CVENGINE_URI'])
        self.dialogueState = DialogueState(getOntologyByName(config['ONTOLOGY']))

        # Storing stuff
        self.observation = {}

    def name(self):
        return self.__class__.__name__

    def reset(self):
        """ Resets session
        """
        self.to_start_session()  # Check on_enter_ask_ier for more ops

        self.observation = {}

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
        self.intent_slots = []
        self.query_slots = []

    def clear_mask(self):
        self.mask_str_slots = []

    def observe(self, observation):
        """ Updates observation and clears actions
        """
        self.observation.update(observation)

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

        self.observation = {}

    ########################################
    #   Different act according to states  #
    ########################################
    def observe_apply_return(method):
        """
            Observes user act, applies method, then builds return object
            Decorates state_actions that needs to return a system act
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
        """Starts a session by opening an image. 
            Goes through state transition
        """
        assert len(user_acts) == 1
        user_act = user_acts[0]
        assert user_act['dialogue_act'] in ["open", "load"]
        intent = user_act['intent']
        slots = user_act['slots']

        system_acts = []

        self.greeting()  # start_session -> ask_ier

        self.update_slots(intent, slots)

        self.highNLConf_noMissing()  # ask_ier -> ier_complete

        self.noMaskMissing()  # ier_complete -> execute_api

        system_acts += self.act_execute()  # action in execute_api

        self.result()  # execute_api -> ask_ier

        system_acts += sys_act_builder('greeting')
        system_acts += sys_act_builder('ask')

        return system_acts, False

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
            system_acts += self.state_transition()
        elif dialogue_act == "bye":
            self.bye()
            system_acts += sys_act_builder('bye')
            episode_done = True
        else:
            raise ValueError("Invalid dialogue_act: {}".format(dialogue_act))
        return system_acts, episode_done

    @observe_apply_return
    def act_confirm(self, user_acts):
        """ actions in confirm state
            Possible sys_acts:
            - confirm
            - request
            - query
        """
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
        system_acts = self.state_transition()
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

        self.update_slots(user_act['intent'], user_act['slots'])

        # Build sys_act
        system_acts += self.state_transition()

        return system_acts, False

    def state_transition(self):
        """ Performs state transitions according to slot status
            Returns a sys_act object
        """
        # Build sys_act
        system_acts = []
        if self.hasConfirmSlots():
            self.lowNLConf()
            system_acts += sys_act_builder('confirm', self.confirm_slots[:1])
        elif self.hasRequestSlots():
            self.highNLConf_missing()
            system_acts += sys_act_builder('request', self.request_slots)
        elif len(self.query_slots) == 0:  # Nothing learned
            self.zeroNLConf()
            system_acts += sys_act_builder('repeat')
        else:
            self.highNLConf_noMissing()  # -> ier_complete

            # Check if object needs to be queried
            mask_missing = self.act_ier_complete()
            if mask_missing:
                self.maskMissing()
                high_cv_conf = self.act_query()
                if high_cv_conf:
                    self.highCVConf()
                    self.noMaskMissing()
                    system_acts += self.act_execute()
                    self.result()
                    system_acts += sys_act_builder('ask')
                else:
                    self.lowCVConf()
                    if len(self.mask_str_slots) == 0:
                        # Treat this case as ood currently
                        self.noLabel()
                        system_acts += sys_act_builder('ood_cv')
                        system_acts += sys_act_builder('ask')
                    else:
                        system_acts += sys_act_builder(
                            'request_label', slots=self.mask_str_slots)
            else:
                self.noMaskMissing()
                system_acts += self.act_execute()
                self.result()
                system_acts += sys_act_builder('ask')

        return system_acts

    def update_slots(self, intent, inform_slots):
        """ Updates request, confirm, query slots
        """
        query_slot_names = Ontology.getArgumentsWithIntent(intent)

        # Update confirm, request, query slots
        for query_name in query_slot_names:
            inform_idx, slot = find_slot_with_key(
                query_name, inform_slots)
            if inform_idx < 0:
                if query_name not in self.request_slots and \
                        not any(query_name == d['slot'] for d in self.confirm_slots) and \
                        not any(query_name == d['slot'] for d in self.query_slots):
                    self.request_slots.append(query_name)
            else:
                # Found in confirm_slots
                if any(query_name == d['slot'] for d in self.confirm_slots):
                    confirm_idx, _ = find_slot_with_key(
                        query_name, self.confirm_slots)
                    self.confirm_slots.pop(confirm_idx)
                # Found in request_slots
                if query_name in self.request_slots:
                    self.request_slots.remove(query_name)

                if slot['conf'] >= 0.8:
                    self.query_slots.append(slot)
                else:
                    self.confirm_slots.append(slot)

    def hasConfirmSlots(self):
        return len(self.confirm_slots) != 0

    def hasRequestSlots(self):
        return len(self.request_slots) != 0

    ###########################
    #       CV Related        #
    ###########################
    def act_ier_complete(self):
        """action at state ier_complete
           checks if the object slot is present or whether mask is present
        """
        object_idx, object_slot = find_slot_with_key(
            'object', self.query_slots)
        if object_idx < 0 or object_slot['value'] == "image" \
                or len(self.mask_str_slots) == 1:
            # No object is needed, whole image or already has mask
            return False
        return True

    def act_query(self):
        """queries cv engine to obtain masks
            return True for high CV conf, False for low conf
        """
        b64_img_str = self.observation.get('b64_img_str')

        _, object_slot = find_slot_with_key('object', self.query_slots)
        noun = object_slot['value']

        masks = CVEngineAPI.select_object(noun, b64_img_str)

        self.mask_str_slots.clear()
        for idx, mask_str in enumerate(masks):
            mask_str_slot = {'slot': str(idx), 'value': mask_str}
            self.mask_str_slots.append(mask_str_slot)

        return len(self.mask_str_slots) == 1

    @observe_apply_return
    def act_ask_user_label(self, user_acts):
        """action at the state ask_user_label
        """
        system_acts = []
        for user_act in user_acts:
            user_dialogue_act = user_act['dialogue_act']
            if user_dialogue_act == "negate":
                self.noLabel()  # ask_user_label -> ask_ier, clears_ier
                self.clear_mask()
                system_acts += sys_act_builder('clear_edit')

            elif user_dialogue_act == "inform":
                self.hasLabel()  # ask_user_label -> ier_complete

                intent_idx, intent_slot = find_slot_with_key(
                    'intent', user_act['slots'])
                assert intent_idx >= 0 and intent_slot['value'] == "select_object_mask_id"

                mask_id_slots = user_act['slots']

                _, object_mask_id_slot = find_slot_with_key(
                    'object_mask_id', mask_id_slots)

                assert object_mask_id_slot['slot'] == "object_mask_id"
                mask_id = object_mask_id_slot['value']
                assert isinstance(mask_id, str)

                _, selected_mask_slot = find_slot_with_key(
                    mask_id, self.mask_str_slots)
                self.mask_str_slots = [selected_mask_slot]

                intent_idx, intent_slot = find_slot_with_key(
                    'intent', self.query_slots)
                if intent_idx >= 0:
                    self.query_slots.pop(intent_idx)

                self.query_slots += mask_id_slots

                self.noMaskMissing()  # ier_complete -> execute_api

                system_acts += self.act_execute()

                self.result()

                system_acts += sys_act_builder('ask')
            else:
                raise ValueError(
                    "Invalid user_dialogue_act: {}".format(user_dialogue_act))

        return system_acts, False

    def act_execute(self):
        """ action at the state execute_api
            Interesting thing is, since Photoshop is also an agent in the environment
            the only thing we need to do here is to create a sys_act object
        """

        # Execute Intent

        sys_act = sys_act_builder(
            'execute', self.query_slots + self.mask_str_slots)
        return sys_act

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
            elif sys_act['dialogue_act'] == 'ood_nl':
                utt = "Sorry, Photoshop currently does not support this function."
            elif sys_act['dialogue_act'] == 'ood_cv':
                utt = "Sorry, Photoshop's CV engine cannot find what you are looking for."
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
                mask_str_slots = sys_act['slots']
                if len(mask_str_slots) == 0:
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
