import logging
import sys

import numpy as np

from ..core import SystemAct
from ..util import load_from_json, slots_to_args, build_slot_dict, find_slot_with_key

from .drl import QNetwork, ReplayMemory, tf_utils
from .drl.scheduler import builder as schedulerlib

logger = logging.getLogger(__name__)


def build_sys_act(dialogue_act, intent=None, slots=None):
    sys_act = {}
    sys_act['dialogue_act'] = build_slot_dict("dialogue_act", dialogue_act)
    if intent is not None:
        sys_act['intent'] = build_slot_dict("intent", intent)
    if slots is not None:
        sys_act['slots'] = slots
    return sys_act


class ActionMapper(object):
    """
    Maps index to user_act with ontology_json
    """

    def __init__(self, ontology_json):
        """
        Builds action map here
        """
        action_map = {}

        # Request
        for slot in ontology_json["slots"]:
            action_info = {
                'dialogue_act': SystemAct.REQUEST,
                'slot': slot["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Confirm
        for slot in ontology_json["slots"]:
            action_info = {
                'dialogue_act': SystemAct.CONFIRM,
                'slot': slot["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Query
        action_idx = len(action_map)
        action_map[action_idx] = {
            'dialogue_act': SystemAct.QUERY,
            'slot': 'object'
        }

        # Execute
        for intent in ontology_json["intents"]:
            action_info = {
                'dialogue_act': SystemAct.EXECUTE,
                'intent': intent["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        self.action_map = action_map

    def size(self):
        return len(self.action_map)

    def __call__(self, action_idx, state):
        """
        Process action_idx into sys_act object
        """
        action_dict = self.action_map[action_idx]
        da = action_dict['dialogue_act']
        if da == SystemAct.REQUEST:
            intent = da
            slot_name = action_dict['slot']
            slot_dict = build_slot_dict(slot_name)
            slots = [slot_dict]
        elif da in [SystemAct.CONFIRM, SystemAct.QUERY]:
            intent = da
            slot_name = action_dict['slot']
            slot_dict = state.get_slot(slot_name).to_json()
            slots = [slot_dict]
        elif da == SystemAct.EXECUTE:
            intent = action_dict['intent']
            intent_node = state.get_intent(intent)

            slots = []
            for child_node in intent_node.children.values():
                slot_dict = child_node.to_json()
                slots.append(slot_dict)
        else:
            raise ValueError("Unknown dialogue act: {}".format(da))

        sys_act = build_sys_act(da, intent, slots)
        return sys_act


class BasePolicy(object):
    """
    Base class for all policies, also records reward

    public methods:
        next_action
        record
    private methods:
        build_action_map
        step
        action_idx_to_sys_act

    Attributes:
        state_size (int):
        action_size (int):
        action_map (dict): map action index to system actions
    """

    def __init__(self, state_size, action_mapper, **kwargs):
        self.state_size = state_size
        self.action_mapper = action_mapper
        self.action_size = self.action_mapper.size()

    def reset(self):
        self.rewards = []

    def next_action(self, state):
        """
        Args:
            state (object): state of the system
        Returns:
            sys_act (dict): one system_action
        """
        # Predict next action index
        state_list = state.to_list()
        action_idx = self.step(state_list)
        sys_act = self.action_mapper(action_idx, state)
        return sys_act

    def step(self, state):
        """
        Override this class for customized policies
        Args:
            state (list): list of float
        returns:
            action (int): action index
        """
        raise NotImplementedError

    def add_reward(self, reward):
        """
        Records 
        """
        self.rewards.append(reward)


class RandomPolicy(BasePolicy):
    """
    Picks an action randomly
    """

    def step(self, state):
        """
        Much as a gym agent step, take a vector as state input
        Args:
            state (list): list of floats
        Returns:
            action (int): action index
        """
        return np.random.randint(0, self.action_size)


class CommandLinePolicy(BasePolicy):
    """
    Policy interaction via command line
    """

    def step(self, state):
        """
        Display action index and corresponding action descriptions
        and ask for integer input
        """
        cmd_msg_list = []
        for action_idx, action_dict in self.action_map.items():
            da_name = action_dict.get('dialogue_act')
            slot_name = action_dict.get("intent") or action_dict.get("slot")
            action_msg = "Action {} : {} {}".format(
                action_idx, da_name, slot_name)
            cmd_msg_list.append(action_msg)

        cmd_msg = " | ".join(cmd_msg_list)
        action_index = -1

        while action_index < 0 or action_index >= self.action_size:
            print(cmd_msg)
            action_index = int(
                input("[CMDPolicy] Please input an action index: "))

        return action_index


class RulePolicy(BasePolicy):
    """
    Rule based policy using sysintent pulling
    """

    def next_action(self, state):
        """
        A simple rule-based policy
        Args:
            state (object): State defined in state.py
            reward (float): reward resulting from previous action
        Returns:
            sys_act (list): list of sys_acts
        """
        sysintent = state.pull()

        if len(sysintent.confirm_slots):
            da = SystemAct.CONFIRM
            intent = da
            slots = sysintent.confirm_slots[:1].copy()  # confirm 1 at a time

        elif len(sysintent.request_slots):
            da = SystemAct.REQUEST
            intent = da
            slots = sysintent.request_slots[:1].copy()  # request 1 at a time

        elif len(sysintent.query_slots):
            da = SystemAct.QUERY
            intent = da
            slots = sysintent.query_slots.copy()

        else:
            da = SystemAct.EXECUTE
            intent = state.get_slot('intent').get_max_value()
            slots = sysintent.execute_slots.copy()

        sys_act = build_sys_act(da, intent, slots)
        return sys_act


class DQNPolicy(BasePolicy):
    """
    Base class for all policies, also records reward

    public methods:
        next_action
        record
    private methods:
        build_action_map
        step
        action_idx_to_sys_act

    Attributes:
        config: 
        action_mapper:
    """

    def __init__(self, policy_config, action_mapper):
        # Setup config
        self.config = policy_config
        self.action_mapper = action_mapper
        self.config["qnetwork"]["output_size"] = self.action_mapper.size()

        # Load configuration
        self.build_from_config()

        # Create session and saver
        self.sess = tf_utils.create_session()
        self.saver = tf_utils.create_saver()

        # Initialize all variables
        tf_utils.initialize_all_variables(self.sess)

    def build_from_config(self):

        # Directly load config
        self.batch_size = self.config["batch_size"]
        self.epsilon = float(self.config["scheduler"]['init_epsilon'])

        # Create Q Network and Target Q Network
        source_name = 'qnetwork'
        target_name = 'target_qnetwork'
        qnetwork_config = self.config["qnetwork"]
        self.qnetwork = QNetwork(qnetwork_config, name=source_name)
        self.target_qnetwork = QNetwork(qnetwork_config, name=target_name)
        self.copy_op = tf_utils.copy_variable_scope(source_name, target_name)

        # Replay Memory
        self.replaymemory = ReplayMemory(**self.config["replaymemory"])

        # Scheduler
        scheduler_name = self.config["scheduler"]["scheduler"]
        self.scheduler = schedulerlib(scheduler_name)(
            **self.config["scheduler"])

    def reset(self):
        self.rewards = []

        self.previous_state = None
        self.previous_action = None
        self.state = None
        self.action = None
        self.reward = None
        self.episode_done = None

    def next_action(self, state):
        """
        Args:
            state (object): state of the system
        Returns:
            sys_act (dict): one system_action
        """
        # Predict next action index
        state_list = state.to_list()
        action_idx = self.step(state_list)
        sys_act = self.action_mapper(action_idx, state)

        self.previous_state = self.state
        self.previous_action = self.action

        self.state = state_list
        self.action = action_idx
        return sys_act

    def step(self, state):
        """
        Args:
        - state: list

        Return: 
        - action: int
        """
        if isinstance(state, list):
            assert not any(isinstance(x, list) for x in state)

        state = np.expand_dims(state, axis=0)  # Expect only one dimension only

        q_values = self.qnetwork.predict_batch(
            self.sess, state)  # (batch_size, action_space)

        # Here we decide which epsilon decay policy we should use
        action = self.epsilon_greedy_policy(q_values)

        return action

    def record(self, reward, episode_done):
        """
        Sends into replay memory
        """
        self.reward = reward
        self.episode_done = episode_done

        if self.previous_state:
            self.replaymemory.add(
                self.previous_state, self.previous_action, self.reward, self.state, self.episode_done)

    ########################
    #  Tensorflow Related  #
    ########################
    def copy_qnetwork(self):
        # Copy current Qnetwork to previous qnetwork
        self.sess.run([self.copy_op])

    def update_epsilon(self, current_timestep=None, test=False):
        if not test:
            self.epsilon = self.scheduler.value(current_timestep)
        else:
            self.epsilon = self.scheduler.end_value()
        return self.epsilon

    def epsilon_greedy_policy(self, qvalues, epsilon=None):
        """
        Sample according to probability
        """
        action_size = qvalues.size

        max_action_index = np.argmax(qvalues)
        probs = []
        for action_index in range(action_size):
            if action_index == max_action_index:
                action_prob = 1 - self.epsilon
            else:
                action_prob = self.epsilon / (action_size - 1)
            probs.append(action_prob)

        action = np.random.choice(action_size, p=probs)
        return action

    def update_network(self):
        """
        Sample from Replay Memory and update network for one batch
        """
        if self.replaymemory.size() < self.batch_size:
            return 0.0

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = \
            self.replaymemory.sample_encode(self.batch_size)

        # Target QNetwork Prediction
        batch_target_qvalues = self.target_qnetwork.predict_batch(
            self.sess, batch_next_states)  # (batch_size, action_size)

        batch_max_target_qvalues = np.max(batch_target_qvalues, axis=-1)

        batch_target_qvalues = batch_rewards + \
            (1 - batch_done) * batch_max_target_qvalues

        # Pass to QNetwork for update
        batch_loss = self.qnetwork.train_batch(
            self.sess, batch_states, batch_actions, batch_target_qvalues)
        return batch_loss

    def save(self, exp_path, global_step=None):
        """
        Save 
        """
        self.saver.save(self.sess, exp_path, global_step)

    def load(self, load_path):
        """
        Load existing session
        """
        print("[DQNPolicy] Restoring from {}".format(load_path))
        self.saver.restore(self.sess, load_path)


def builder(string):
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError as e:
        print(e)
        logger.error("Unknown policy: {}".format(string))
        return None
