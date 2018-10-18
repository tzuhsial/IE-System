import logging

from ..core import SystemAct
from ..state import State
from ..policy import builder as policylib, ActionMapper
from ..visionengine import VisionEnginePortal
from ..util import find_slot_with_key, build_slot_dict, slots_to_args, load_from_json

logger = logging.getLogger(__name__)


def SystemPortal(system_config):
    ontology_json = load_from_json(system_config['ontology'])
    state = State(ontology_json)
    visionengine = VisionEnginePortal(system_config['visionengine'])

    # Setup Policy here
    policy_config = system_config["policy"]
    # Build action mapper
    action_config = policy_config["action"]
    action_mapper = ActionMapper(action_config)

    policy_config["state_size"] = len(state.to_list())
    policy_config["action_size"] = action_mapper.size()

    print("state_size", policy_config["state_size"])
    print("action_size", policy_config["action_size"])
    policy_name = policy_config["name"]
    policy = policylib(policy_name)(
        policy_config,
        action_mapper,
        ontology_json=ontology_json,
        dialogue_state=state)

    system = System(state, policy, visionengine)
    return system


class System(object):
    """
    Mainly handles the interactions with the environment
    Also provides a rule-based policy

    Attributes:
        state (object)
        policy (object)
        visionengine (object)
    """

    def __init__(self, state, policy, visionengine):
        # Components

        self.state = state
        self.policy = policy
        self.visionengine = visionengine

    def load_policy(self, policy):
        self.policy = policy

    def reset(self):
        """ 
        Resets for a new dialogue session
        """
        self.observation = {}
        self.state.reset()
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
        # Get acts from photoshop & user
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

        # Update turn_id
        self.turn_id += 1
        self.state.turn_id += 1

    def post_policy(self, system_acts):
        """
        Preprocess system_acts into 
        Args:
            system_acts (list): list of system acts
        Returns:
            system_act 
        """
        sys_act = system_acts[0]

        sys_dialogue_act = sys_act['dialogue_act']['value']
        if sys_dialogue_act == SystemAct.CONFIRM:
            confirm_slots = sys_act["slots"]
            self.state.sysintent.confirm_slots = confirm_slots

        elif sys_dialogue_act == SystemAct.QUERY:
            query_slots = sys_act['slots']
            self.query_visionengine(query_slots)

        elif sys_dialogue_act == SystemAct.EXECUTE:
            # Stack the intent to history and clear slots
            intent = sys_act['intent']['value']

        # Build Return object
        system_act = {}
        system_act['system_acts'] = system_acts
        system_act['system_utterance'] = self.template_nlg(system_acts)
        system_act['episode_done'] = self.observation.get(
            'episode_done', False)
        return system_act

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

        ######################
        #     Post Policy    #
        ######################
        system_act = self.post_policy(system_acts)

        return system_act

    def query_visionengine(self, query_slots):
        """
        Query vision engine
        """
        args = slots_to_args(query_slots)
        b64_img_str = self.state.get_slot(
            'original_b64_img_str').get_max_value()
        args['b64_img_str'] = b64_img_str

        mask_strs = self.visionengine.select_object(**args)

        # Postprocess with gesture_click
        object_mask_str_node = self.state.get_slot('object_mask_str')

        # clear_object
        # Force directly add into object_mask_strs
        object_mask_str_node.value_conf_map.clear()
        if len(mask_strs) > 0:
            object_mask_str_node.value_conf_map = \
                {mask_str: 0.5 for mask_str in mask_strs}
        object_mask_str_node.last_update_turn_id += 1
        #print("Query results:", len(mask_strs))

        # Filter candidates with gesture_click
        gesture_click = self.state.get_slot('gesture_click').get_max_value()
        if gesture_click is not None:
            object_mask_str_node.filter_candidates(gesture_click)
            #print("Filtered results", len(object_mask_str_node.value_conf_map))

        object_mask_str_node.last_update_turn_id += 1

    def query_executionhistory(self, query_slots):
        """
        Queries execution history with execution slots
        """
        raise NotImplementedError

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
                req_slot = sys_act['slots'][0]
                req_name = req_slot['slot']
                utt = "What {} do you want?".format(req_name)
            elif sys_dialogue_act == SystemAct.CONFIRM:
                confirm_slots = sys_act['slots']
                utt = "Let me confirm. "
                confirm_list = []
                for slot_dict in confirm_slots:
                    slot_value = str(slot_dict.get('value', ""))
                    if slot_dict['slot'] in [
                            "object_mask_str", "gesture_click",
                            "original_b64_img_str"
                    ]:
                        sv = slot_dict['slot'] + " is " + slot_value[:5]
                    elif slot_dict["slot"] == "mask_strs":
                        sv = slot_dict["slot"] + " is " + str(len(slot_value))
                    else:
                        sv = slot_dict['slot'] + " is " + slot_value
                    confirm_list.append(sv)
                utt += ','.join(confirm_list) + "?"
            elif sys_dialogue_act == SystemAct.QUERY:
                query_slots = sys_act['slots']

                slot_list = []

                for slot_dict in query_slots:
                    slot_value = str(slot_dict.get('value', ""))
                    if slot_dict['slot'] in [
                            "object_mask_str", "gesture_click"
                    ]:
                        sv = slot_dict['slot'] + " = " + slot_value[:5]
                    elif slot_dict["slot"] == "mask_strs":
                        sv = slot_dict["slot"] + "=" + str(len(slot_value))
                    else:
                        sv = slot_dict['slot'] + "=" + slot_value
                    slot_list.append(sv)

                utt = "Query vision engine with " + ', '.join(slot_list) + "."
            elif sys_dialogue_act == SystemAct.EXECUTE:
                execute_slots = sys_act['slots'] + [sys_act['intent']]

                slot_list = []
                for slot in execute_slots:
                    slot_value = str(slot.get('value', ""))
                    if slot["slot"] == "object_mask_str":
                        sv = slot['slot'] + "=" + slot_value[:5]
                    elif slot["slot"] == "mask_strs":
                        sv = slot["slot"] + "=" + str(len(slot_value))
                    else:
                        sv = slot['slot'] + "=" + slot_value
                    slot_list.append(sv)
                utt = "Execute: " + ', '.join(slot_list) + "."
            elif sys_dialogue_act == SystemAct.BYE:
                utt = "Goodbye! See you next time!"
            else:
                import pdb
                pdb.set_trace()
                utt = ""
            utt_list.append(utt)

        full_utterance = ' '.join(utt_list)
        return full_utterance
