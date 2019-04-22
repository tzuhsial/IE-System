import logging

from ..core import SystemAct
from ..state import State
from ..policy import builder as policylib, ActionMapper
from ..util import find_slot_with_key, build_slot_dict, slots_to_args, load_from_json

logger = logging.getLogger(__name__)


def ManagerPortal(manager_config):
    # Setup state
    ontology_json = load_from_json(manager_config['state']['ontology'])
    state = State(ontology_json)

    # Setup Policy here
    policy_config = manager_config["policy"]
    # Build action mapper
    action_config = policy_config["action"]
    action_mapper = ActionMapper(action_config)

    policy_config["state_size"] = len(state.to_list())
    policy_config["action_size"] = action_mapper.size()

    #print("state_size", policy_config["state_size"])
    #print("action_size", policy_config["action_size"])
    policy_name = policy_config["name"]
    policy = policylib(policy_name)(
        policy_config,
        action_mapper,
        ontology_json=ontology_json,
        dialogue_state=state)

    manager = DialogueManager(state, policy)
    return manager


class DialogueManager(object):
    """
    Mainly handles the interactions with the environment
    Also provides a rule-based policy

    Attributes:
        state (object)
        policy (object)
        visionengine (object)
    """

    def __init__(self, state, policy):
        # Components
        self.state = state
        self.policy = policy

    def load_policy(self, policy):
        self.policy = policy

    def reset(self):
        """ 
        Resets for a new dialogue session
        """
        self.observation = {}
        self.state.reset()
        self.policy.reset()
        self.turn_id = 1

    def flush(self):
        """ Flush state values 
        """
        self.observation = {}
        self.state.flush()

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
        # Get acts from imageeditengine & user & visionengine
        imageeditengine_acts = self.observation.get(
            'imageeditengine_acts', list())
        user_acts = self.observation.get('user_acts', list())
        visionengine_act = self.observation.get('visionengine_acts', list())
        observed_acts = imageeditengine_acts + user_acts + visionengine_act

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
            # Record the previous confirmed slots
            confirm_slots = sys_act["slots"]
            self.state.sysintent.confirm_slots = confirm_slots

        elif sys_dialogue_act == SystemAct.QUERY:
            # Do nothing here
            pass

        elif sys_dialogue_act == SystemAct.EXECUTE:
            # Do nothing here too
            pass

        # Always add another action to give image edit engine the current mask
        sys2ie_slots = []
        mask_node = self.state.get_slot('object_mask_str')
        if mask_node.get_max_conf() >= 0.5:
            value, conf = mask_node.get_max_conf_value()
            mask_slot = build_slot_dict('object_mask_str', value, conf)
        else:
            mask_slot = build_slot_dict('object_mask_str')
        sys2ie_slots.append(mask_slot)

        sys2ie_act = {
            'dialogue_act': build_slot_dict('dialogue_act', SystemAct.INFORM, 1.0),
            'slots': sys2ie_slots
        }
        system_acts.append(sys2ie_act)

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

        # Clear Observation
        self.observation = {}

        return system_act

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
                if req_name == "adjust_value":
                    utt = "What value would you like to adjust?"
                else:
                    utt = "What {} do you want?".format(req_name)
            elif sys_dialogue_act == SystemAct.CONFIRM:
                confirm_slots = sys_act['slots']
                utt = ""
                assert len(confirm_slots) == 1
                slot_dict = confirm_slots[0]
                slot_name = slot_dict["slot"]
                slot_value = str(slot_dict.get('value', ""))
                if slot_dict['slot'] in ["object_mask_str"]:
                    utt = "Is the current detected object region correct?"
                else:
                    utt = "Is your {} {}?".format(slot_name, slot_value)
                utt += " (yes/no)"
            elif sys_dialogue_act == SystemAct.QUERY:
                """
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
                """
                utt = ""
                slot_list = []
                for slot_dict in sys_act['slots']:
                    slot_value = str(slot_dict.get('value', ""))
                    if slot_dict['slot'] == "object":
                        utt += "Sending \"{}\" to vision engine."\
                            .format(slot_value)

            elif sys_dialogue_act == SystemAct.EXECUTE:
                utt = "Executed image edit."
            elif sys_dialogue_act == SystemAct.BYE:
                utt = "Goodbye! See you next time!"
            else:
                continue
            utt_list.append(utt)

        full_utterance = ' '.join(utt_list)
        return full_utterance

    def to_json(self):
        obj = {
            "state": self.state.to_json(),
            "policy": self.policy.to_json(),
            "turn_id": self.turn_id
        }
        return obj

    def from_json(self, obj):
        self.state.from_json(obj['state'])
        self.policy.from_json(obj['policy'])
        self.turn_id = obj['turn_id']
