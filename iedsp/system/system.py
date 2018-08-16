from ..core import SystemAct
from ..policy import builder as policylib
from .state import StatePortal
from ..visionengine import VisionEnginePortal
from ..util import find_slot_with_key, build_slot_dict, slots_to_args, load_from_json


class System(object):
    """
    Mainly handles the interactions with the environment
    Also provides a rule-based policy

    Attributes:
        state (object)
        visionengine (object)
        policy (object)
        action_map (dict)
    """

    def __init__(self, global_config):
        # Components
        self.state = StatePortal(global_config)
        self.visionengine = VisionEnginePortal(global_config["VISIONENGINE"])

        # Build policy
        ontology_json = load_from_json(
            global_config["DEFAULT"]["ONTOLOGY_FILE"])
        policy_name = global_config["SYSTEM"]["POLICY"]
        state_size = len(self.state.to_list())

        self.policy = policylib(policy_name)(state_size, ontology_json)

    def reset(self):
        """ 
        Resets for a new dialogue session
        """
        self.observation = {}
        self.state.clear()
        self.policy.reset()
        self.turn_id = 0

    def observe(self, observation):
        """
        If has only utterance, send to tracker for dialogue_act & slots 
        Args:
            observation (dict): observation given by user passed through channel
        """
        self.observation.update(observation)

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

    def act(self):
        """ 
        Perform action according to observation
        Returns:
            system_act (dict)
        """
        ####################
        #   State Update   #
        ####################
        self.state_update()

        ####################
        #      Policy      #
        ####################
        sys_act = self.policy.next_action(self.state)
        system_acts = [sys_act]

        reward = self.observation.get('reward', 0.0)
        #print('[System] reward', reward)
        self.policy.add_reward(reward)

        # Part of the environment
        sys_dialogue_act = sys_act['dialogue_act']['value']
        if sys_dialogue_act == SystemAct.QUERY:
            query_slots = sys_act['slots']
            self.query_visionengine(query_slots)
        elif sys_dialogue_act == SystemAct.EXECUTE:
            # Stack the intent to history and clear slots
            intent = sys_act['intent']['value']
            self.state.stack_intent(intent)
            self.state.clear_ontology()

        # Update turn_id
        self.turn_id += 1

        # Build Return object
        system_act = {}
        system_act['system_acts'] = system_acts
        system_act['system_utterance'] = self.template_nlg(system_acts)
        system_act['episode_done'] = self.observation.get(
            'episode_done', False)

        return system_act

    def query_visionengine(self, query_slots):
        """
        Query visionengine with query_slots
        """
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

        object_mask_str_node.last_update_turn_id += 1

    def get_reward(self):
        rewards = self.policy.rewards
        return sum(rewards)

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
                    slot_value = str(slot_dict.get('value', ""))
                    if slot_dict['slot'] == "object_mask_str":
                        sv = slot_dict['slot'] + " is " + slot_value[:5]
                    else:
                        sv = slot_dict['slot'] + " is " + slot_value
                    confirm_list.append(sv)
                utt += ','.join(confirm_list) + "?"
            elif sys_dialogue_act == SystemAct.QUERY:
                query_slots = sys_act['slots']
                slot_list = [slot['slot'] + "=" + str(slot.get('value', ""))
                             for slot in query_slots]
                utt = "Query Vision Engine: " + ', '.join(slot_list) + "."
            elif sys_dialogue_act == SystemAct.EXECUTE:
                execute_slots = sys_act['slots'] + [sys_act['intent']]

                slot_list = []
                for slot in execute_slots:
                    slot_value = str(slot.get('value', ""))
                    if slot["slot"] == "object_mask_str":
                        sv = slot['slot'] + "=" + slot_value[:5]
                    else:
                        sv = slot['slot'] + "=" + slot_value
                    slot_list.append(sv)
                utt = "Execute: " + ', '.join(slot_list) + "."
            elif sys_dialogue_act == SystemAct.BYE:
                utt = "Goodbye! See you next time!"
            else:
                utt = ""
            utt_list.append(utt)

        full_utterance = ' '.join(utt_list)
        return full_utterance
