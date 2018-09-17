import logging
import sys

import numpy as np

from ..core import SystemAct
from ..util import load_from_json, slots_to_args, build_slot_dict, find_slot_with_key

from .drl import ReplayMemory, tf_utils
from .drl import models as modellib
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

    def __init__(self, ontology_json, action_config):
        """
        Builds action map here
        """
        self.config = action_config
        action_map = {}

        # Request
        for slot in ontology_json["slots"]:
            if slot["name"] not in action_config["slots"]:
                continue
            action_info = {
                'dialogue_act': SystemAct.REQUEST,
                'slot': slot["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Confirm
        for slot in ontology_json["slots"]:
            if slot["name"] not in action_config["slots"]:
                continue
            action_info = {
                'dialogue_act': SystemAct.CONFIRM,
                'slot': slot["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Query
        for slot in ontology_json["slots"]:
            if slot["name"] not in action_config["query"]:
                continue

            action_info = {
                'dialogue_act': SystemAct.QUERY,
                'slot': slot["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        # Execute
        for intent in ontology_json["intents"]:
            if intent["name"] not in action_config["intents"]:
                continue
            action_info = {
                'dialogue_act': SystemAct.EXECUTE,
                'intent': intent["name"]
            }
            action_idx = len(action_map)
            action_map[action_idx] = action_info

        self.action_map = action_map

        # Build inverse map here
        self.inv_action_map = {}
        self.inv_action_map[SystemAct.REQUEST] = {}
        self.inv_action_map[SystemAct.CONFIRM] = {}
        self.inv_action_map[SystemAct.QUERY] = {}
        self.inv_action_map[SystemAct.EXECUTE] = {}

        for action_idx, action_info in self.action_map.items():
            da = action_info['dialogue_act']
            if da != SystemAct.EXECUTE:
                slot_name = action_info['slot']
                self.inv_action_map[da][slot_name] = action_idx
            else:
                intent_name = action_info['intent']
                self.inv_action_map[da][intent_name] = action_idx

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

    def sys_act_to_action_idx(self, sys_act):
        da = sys_act["dialogue_act"]['value']
        intent = sys_act["intent"]
        slots = sys_act["slots"]

        if da != SystemAct.EXECUTE:
            key = slots[0]['slot']
        else:
            key = intent['value']
        action_idx = self.inv_action_map[da][key]
        return action_idx


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
        replaymemory (object): records (state, action, reward, next_action, episode_done)
    """

    def __init__(self, policy_config, action_mapper, **kwargs):
        self.config = policy_config
        self.state_size = policy_config["state_size"]
        self.action_size = policy_config["action_size"]
        self.action_mapper = action_mapper

        self.replaymemory = ReplayMemory(**policy_config["replaymemory"])

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

        # Feature size
        self.previous_state = self.state
        self.previous_action = self.action
        self.state = state_list
        self.action = action_idx
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

    def record(self, reward, episode_done):
        """
        Sends into replay memory
        """
        self.reward = reward
        self.episode_done = episode_done

        if self.previous_state:  # is not None
            self.replaymemory.add(self.previous_state, self.previous_action,
                                  self.reward, self.state, self.episode_done)


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
        for action_idx, action_dict in self.action_mapper.action_map.items():
            da_name = action_dict.get('dialogue_act')
            slot_name = action_dict.get("intent") or action_dict.get("slot")
            action_msg = "Action {} : {} {}".format(action_idx, da_name,
                                                    slot_name)
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

        action_idx = self.action_mapper.sys_act_to_action_idx(sys_act)

        # Find action index from sys_act
        state_list = state.to_list()

        self.previous_state = self.state
        self.previous_action = self.action
        self.state = state_list
        self.action = action_idx

        return sys_act


class DQNPolicy(BasePolicy):
    """
    Base class for all policies, also records reward

    public methods:
        next_action
        record
    private methods:
        step

    Attributes:
        config: 
        action_mapper:
    """

    def __init__(self, policy_config, action_mapper, **kwargs):
        # Setup config
        self.config = policy_config
        self.action_mapper = action_mapper
        self.qnetwork_prefix = kwargs.get("qnetwork_prefix", "")

        self.config["qnetwork"]["input_size"] = policy_config["state_size"]
        self.config["qnetwork"]["output_size"] = policy_config["action_size"]

        # Build with configuration
        self.build_from_config()

        # Create session and saver
        self.sess = tf_utils.create_session()
        self.saver = tf_utils.create_saver()

        # Initialize all variables
        tf_utils.initialize_all_variables(self.sess)

        # Log to tensorboard
        logdir = self.config["logdir"]
        self.writer = tf_utils.create_filewriter(logdir, self.sess.graph)

    def build_from_config(self):

        # Directly load config
        self.batch_size = self.config["batch_size"]
        self.epsilon = float(self.config["scheduler"]['init_epsilon'])
        self.gamma = float(self.config["gamma"])

        # Create Q Network and Target Q Network
        if self.qnetwork_prefix != "":
            source_name = self.qnetwork_prefix + '_qnetwork'
            target_name = self.qnetwork_prefix + '_target_qnetwork'
        else:
            source_name = "qnetwork"
            target_name = "target_qnetwork"
        qnetwork_config = self.config["qnetwork"]
        self.qnetwork = modellib.QNetwork(qnetwork_config, name=source_name)
        self.target_qnetwork = modellib.QNetwork(
            qnetwork_config, name=target_name)
        self.copy_op = tf_utils.copy_variable_scope(source_name, target_name)

        # Replay Memory
        self.replaymemory = ReplayMemory(**self.config["replaymemory"])

        # Scheduler
        scheduler_name = self.config["scheduler"]["scheduler"]
        self.scheduler = schedulerlib(scheduler_name)(
            **self.config["scheduler"])

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

        if self.previous_state:  # is not None
            self.replaymemory.add(self.previous_state, self.previous_action,
                                  self.reward, self.state, self.episode_done)

    ##########################
    #   Tensorflow Related   #
    ##########################
    def copy_qnetwork(self):
        # Copy current Qnetwork to previous qnetwork
        self.sess.run([self.copy_op])

    def update_epsilon(self, current_timestep=None, test=False):
        if not test:
            self.epsilon = self.scheduler.value(current_timestep)
        else:
            self.epsilon = self.scheduler.end_value()
        return self.epsilon

    def epsilon_greedy_policy(self, qvalues):
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
            self.gamma * (1 - batch_done) * batch_max_target_qvalues

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

    def log_scalar(self, tag, value, step):
        """
        Log scalar to tensorboard
        Args:
            tag (str): name of scalar
            value (float): value
            step (int): number of step
        """
        summary = tf_utils.create_summary_value(tag, value)
        self.writer.add_summary(summary, step)


class A2CPolicy(BasePolicy):
    """
    Implementation of an actor-critic policy
    """

    def __init__(self, policy_config, action_mapper, **kwargs):
        # Setup config
        self.config = policy_config
        self.config["actor"]["input_size"] = policy_config["state_size"]
        self.config["actor"]["output_size"] = policy_config["action_size"]
        self.config["critic"]["input_size"] = policy_config["state_size"]
        self.config["critic"]["output_size"] = 1  # V(S)

        # Build with configuration
        self.build_from_config()

        # Setup action mapper
        self.action_mapper = action_mapper

        # Create session and saver
        self.sess = tf_utils.create_session()
        self.saver = tf_utils.create_saver()

        # Initialize all variables
        tf_utils.initialize_all_variables(self.sess)

        # Log to tensorboard
        logdir = self.config["logdir"]
        self.writer = tf_utils.create_filewriter(logdir, self.sess.graph)

    def build_from_config(self):
        # Gamma
        self.gamma = float(self.config["gamma"])

        # Create Actor and Critic
        self.actor = modellib.Actor(self.config["actor"], name="actor")
        self.critic = modellib.Critic(self.config["critic"], name="critic")

    def reset(self):
        self.rewards = []
        self.states = []
        self.actions = []
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

        # Store state & action
        self.states.append(state_list)
        self.actions.append(action_idx)
        return sys_act

    def step(self, state):
        """
        Args:
            state: list

        Return: 
            action_idx: int
        """
        if isinstance(state, list):
            assert not any(isinstance(x, list) for x in state)

        state = np.expand_dims(state, axis=0)  # Expect only one dimension only
        action_probs = self.actor.predict_batch(self.sess, state)
        action_idx = np.random.choice(
            action_probs.size, p=action_probs.squeeze())
        return action_idx

    def record(self, reward, episode_done):
        """
        Sends into replay memory
        """
        self.rewards.append(reward)
        self.episode_done = episode_done

    ##########################
    #   Tensorflow Related   #
    ##########################
    def compute_reward(self, rewards, gamma):
        Gts = []
        T = len(rewards)

        # Downscale
        rewards = [r / 100 for r in rewards]

        for t in range(T - 1, -1, -1):
            if t == T - 1:
                gt = rewards[t]
            else:
                gt = gamma * Gts[-1] + rewards[t]
            Gts.append(gt)

        Gts = Gts[::-1]
        return Gts

    def end_episode(self, train_mode=False):
        """
        Perform a network update
        Args:
            train_mode (bool)
        Returns:
            actor_batch_loss (float)
            critic_batch_loss (float)
        """
        if not train_mode:
            return 0., 0.

        Gts = self.compute_reward(self.rewards, self.gamma)

        # Preprocess to numpy
        batch_states = np.array(self.states)
        batch_state_values = self.critic.predict_batch(self.sess, batch_states)
        batch_actions = np.array(self.actions)

        batch_Gts = np.array(Gts)
        batch_Gts = np.expand_dims(batch_Gts, axis=1)

        # Advantage function
        batch_advts = batch_Gts - batch_state_values

        # Update actor
        actor_batch_loss = self.actor.train_batch(self.sess, batch_states,
                                                  batch_actions, batch_advts)

        # Update critic
        critic_batch_loss = self.critic.train_batch(self.sess, batch_states,
                                                    batch_Gts)

        return actor_batch_loss, critic_batch_loss

    def save(self, exp_path, global_step=None):
        """
        Save 
        """
        self.saver.save(self.sess, exp_path, global_step)

    def load(self, load_path):
        """
        Load existing session
        """
        print("[A2CPolicy] Restoring from {}".format(load_path))
        self.saver.restore(self.sess, load_path)

    def log_scalar(self, tag, value, step):
        """
        Log scalar to tensorboard
        Args:
            tag (str): name of scalar
            value (float): value
            step (int): number of step
        """
        summary = tf_utils.create_summary_value(tag, value)
        self.writer.add_summary(summary, step)


class HierarchicalPolicy(BasePolicy):
    """
    HierarchicalPolicy is, in fact, meta policy & a set of sub policies
    Attributes:
        meta_controller (DQNPolicy)
        controllers (dict): dict of DQNPolicies with option_idx as key
    """

    def __init__(self, policy_config, action_mapper, **kwargs):
        # Setup config
        self.config = policy_config
        self.action_mapper = action_mapper
        self.ontology_json = kwargs['ontology_json']

        self.build_from_config()

        # Create session and saver
        self.sess = tf_utils.create_session()
        #self.saver = tf_utils.create_saver()

        # Initialize all variables
        tf_utils.initialize_all_variables(self.sess)

        # Log to tensorboard
        #logdir = self.config["logdir"]
        #self.writer = tf_utils.create_filewriter(logdir, self.sess.graph)

    def build_from_config(self):
        # Define options here
        state_size = self.config["state_size"]
        action_size = self.config["action_size"]

        # Build Meta Controller
        meta_controller_config = self.config["meta_controller"]
        option_size = 2  # request(intent), confirm(intent)
        option_size += len(self.action_mapper.config["intents"])
        meta_controller_config["state_size"] = state_size
        meta_controller_config["action_size"] = option_size

        self.meta_controller = DQNPolicy(
            meta_controller_config,
            self.action_mapper,
            qnetwork_prefix="meta_controller")

        # Build primitive action mapping
        request_intent_idx = self.action_mapper\
            .inv_action_map['request']['intent']
        confirm_intent_idx = self.action_mapper\
            .inv_action_map['confirm']['intent']
        self.primitive_actions = {0: request_intent_idx, 1: confirm_intent_idx}

        # Build Controllers
        controller_config = self.config["controller"]
        controller_config["state_size"] = state_size
        controller_config["action_size"] = action_size
        self.controllers = {}
        self.option_execution_map = {}
        for option_idx in range(2, option_size):
            option_name = self.action_mapper.config["intents"][option_idx - 2]

            option_policy = DQNPolicy(
                controller_config,
                self.action_mapper,
                qnetwork_prefix=option_name)
            self.controllers[option_idx] = option_policy
            #print('option_idx', option_idx, option_name)
            option_execute_idx = self.action_mapper.inv_action_map['execute'][
                option_name]
            self.option_execution_map[option_idx] = option_execute_idx

        print("state_size", state_size)
        print("option_size", option_size)
        print("action_size", action_size)

    def meta_step(self, state):
        state = np.expand_dims(state, axis=0)  # Expect 1 dim
        meta_policy = self.meta_controller
        q_values = meta_policy.qnetwork.predict_batch(self.sess, state)
        action = meta_policy.epsilon_greedy_policy(q_values)
        return action

    def option_step(self, option_idx, state):
        state = np.expand_dims(state, axis=0)
        option_policy = self.controllers[option_idx]
        q_values = option_policy.qnetwork.predict_batch(self.sess, state)
        action = option_policy.epsilon_greedy_policy(q_values)
        return action


def builder(string):
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError as e:
        print(e)
        logger.error("Unknown policy: {}".format(string))
        return None
