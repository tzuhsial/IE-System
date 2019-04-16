from .channel import ChannelPortal
from .nlu import NLUPortal
from .user import UserPortal
from .system import SystemPortal
from .photoshop import PhotoshopPortal
from . import util
from .core import SystemAct


class ImageEditWorld(object):
    def __init__(self, config, agents_config, **kwargs):
        """
        Set config and build agents
        Args:
            agents (list):  user 2. channel 3. agent 4. photoshop
        """
        self.config = config

        user = UserPortal(agents_config["user"])
        channel = ChannelPortal(agents_config["channel"])
        system = SystemPortal(agents_config["system"])
        photoshop = PhotoshopPortal(agents_config["photoshop"])

        self.agents = [user, channel, system, photoshop]

    def reset(self):
        """
        Resets for another interaction
        """
        for agent in self.agents:
            agent.reset()

        self.acts = [{}] * len(self.agents)

        self.turn_count = 0

    def parley(self):
        """ 
        Here we define the interaction among agents
        """
        # 0, 1, 2, 3
        user, channel, system, photoshop = self.agents
        user_idx, channel_idx, system_idx, photoshop_idx = 0, 1, 2, 3

        # User observes Agent, Photoshop
        user.observe(self.acts[system_idx])
        user.observe(self.acts[photoshop_idx])
        self.acts[user_idx] = user.act()

        if self.config["verbose"]:
            print("[User]", self.acts[user_idx]['user_utterance'], 'reward',
                  self.acts[user_idx]['reward'])

        # Channel observes User
        channel.observe(self.acts[user_idx])
        self.acts[channel_idx] = channel.act()
        if self.config["verbose"]:
            print("[Channel]", self.acts[channel_idx]['channel_utterance'])

        # Agent observes User from Channel and Photoshop
        system.observe(self.acts[channel_idx])
        system.observe(self.acts[photoshop_idx])
        self.acts[system_idx] = system.act()

        if self.config["verbose"]:
            print("[System]", self.acts[system_idx]['system_utterance'])

        # Photoshop observes agent action
        photoshop.observe(self.acts[system_idx])
        self.acts[photoshop_idx] = photoshop.act()

        self.turn_count += 1

        if self.config["verbose"]:
            print("[World] Turn", self.turn_count)

    def reward(self):
        return self.acts[0]['reward']

    def episode_done(self):
        """
        Return user episode done
        """
        return self.acts[0]['episode_done']


class ImageEditEnvironment(ImageEditWorld):
    """
    Share same APIs like OpenAI gym
    """

    def load_agenda(self, agenda):
        """
        User load agenda
        """
        self.agents[0].load_agenda(agenda)

    def reset(self, agenda=None):
        """
        Resets for another interaction
        Returns:
            state (list): state feature  
        """
        for agent in self.agents:
            agent.reset()

        self.acts = [{}] * len(self.agents)

        self.turn_count = 0

        if agenda is not None:
            self.load_agenda(agenda)

        # Return initial state
        # 0, 1, 2, 3
        user, channel, system, photoshop = self.agents
        user_idx, channel_idx, system_idx, photoshop_idx = 0, 1, 2, 3

        # User observes Agent, Photoshop
        user.observe(self.acts[system_idx])
        user.observe(self.acts[photoshop_idx])
        self.acts[user_idx] = user.act()

        if self.config["verbose"]:
            print("[User]", self.acts[user_idx]['user_utterance'], 'reward',
                  self.acts[user_idx]['reward'])

        # Channel observes User
        channel.observe(self.acts[user_idx])
        self.acts[channel_idx] = channel.act()

        if self.config["verbose"]:
            print("[Channel]", self.acts[channel_idx]['channel_utterance'])

        # Agent observes User from Channel and Photoshop
        system.observe(self.acts[channel_idx])
        system.observe(self.acts[photoshop_idx])

        system.state_update()

        return system.state.to_list()

    def step(self, action_idx):
        """
        Args:
            action_idx (int): action index 
        Returns:
            state (list):
            reward (float)
            done (bool)
            info (dict)
        """

        # 0, 1, 2, 3
        user, channel, system, photoshop = self.agents
        user_idx, channel_idx, system_idx, photoshop_idx = 0, 1, 2, 3

        # Convert action idx into hahahaha
        # An ugly process...
        sys_act = system.policy.action_mapper(action_idx, system.state)
        system_acts = [sys_act]
        self.acts[system_idx] = system.post_policy(system_acts)

        if self.config["verbose"]:
            print("[System]", self.acts[system_idx]['system_utterance'])

        # Photoshop observes agent action
        photoshop.observe(self.acts[system_idx])
        self.acts[photoshop_idx] = photoshop.act()

        if self.config["verbose"]:
            print("[World] Turn", self.turn_count)

        self.turn_count += 1

        # User observes Agent, Photoshop
        user.observe(self.acts[system_idx])
        user.observe(self.acts[photoshop_idx])
        self.acts[user_idx] = user.act()

        if self.config["verbose"]:
            print("[User]", self.acts[user_idx]['user_utterance'], 'reward',
                  self.acts[user_idx]['reward'])

        # Channel observes User
        channel.observe(self.acts[user_idx])
        self.acts[channel_idx] = channel.act()

        if self.config["verbose"]:
            print("[Channel]", self.acts[channel_idx]['channel_utterance'])

        # Agent observes User from Channel and Photoshop
        system.observe(self.acts[channel_idx])
        system.observe(self.acts[photoshop_idx])
        system.state_update()

        state = system.state.to_list()
        reward = self.acts[user_idx]['reward']
        done = self.acts[user_idx]['episode_done']
        info = {}

        return state, reward, done, info


def load_dialoguesystem(args):
    """
    Load Dialogue System
    1. NLU 
    2. System (State + VisionEngine + Dialogue Manager)
    3. Photoshop
    """
    global nlu
    global system
    global photoshop

    config_file = args.config
    config = util.load_from_json(config_file)

    # Load agents here
    agents_config = config["agents"]
    nlu = NLUPortal(agents_config["system"]["nlu"])
    system = SystemPortal(agents_config["system"])
    photoshop = PhotoshopPortal(agents_config["photoshop"])


class ImageEditDialogueSystem(object):
    """
    Interface to real user, can be
    - terminal
    - server

    Attributes
    - nlu
    - system
    - photoshop
    """

    def __init__(self, config_file):
        """
        Set config and build agents
        Args:
            config_file (str)
        """
        # Load config
        config = util.load_from_json(config_file)
        agents_config = config["agents"]

        # Create modules
        self.nlu = NLUPortal(agents_config["system"]["nlu"])
        self.system = SystemPortal(agents_config["system"])
        self.photoshop = PhotoshopPortal(agents_config["photoshop"])

    def reset(self):
        self.nlu.reset()
        self.system.reset()
        self.photoshop.reset()

        self.acts = [{}] * 4  # first one for user

    def open(self, image_path):
        """
        TODO: load b64_img_str
        """
        result, msg = self.photoshop.control(
            "open", {'image_path': image_path})
        assert result

        # Photoshop action
        self.acts[3] = self.photoshop.act()

    def step(self, user_utterance):
        # Order:
        # user, nlu, system, photoshop

        ################
        #   User Act   #
        ################

        # Pass to NLU to get result
        user_act = {"user_utterance": user_utterance}
        self.acts[0] = user_act

        ################
        #    NLU Act   #
        ################
        self.nlu.observe(user_act)
        nlu_act = self.nlu.act()
        self.acts[1] = nlu_act

        ##################
        #   System Act   #
        ##################
        photoshop_act = self.acts[3]
        self.system.observe(photoshop_act)
        self.system.observe(nlu_act)
        system_act = self.system.act()
        self.acts[2] = system_act

        sys_utt = system_act["system_utterance"]

        sys_dialogue_act = system_act['system_acts'][0]['dialogue_act']['value']

        if sys_dialogue_act == SystemAct.QUERY:
            # Interact with Vision Engine
            # Should always return something
            ObjectMaskStrNode = self.system.state.get_slot('object_mask_str')
            assert len(ObjectMaskStrNode.value_conf_map) == 1

            mask_strs = []
            for b64_mask_str in ObjectMaskStrNode.value_conf_map.keys():
                mask_strs.append((1, b64_mask_str))

            # Load mask strs
            result, msg = self.photoshop.control(
                'load_mask_strs', {"mask_strs": mask_strs})
            assert result

            system_act = self.system.act()
            # Update sys_dialogue_act
            sys_dialogue_act = system_act['system_acts'][0]['dialogue_act']['value']
            sys_utt += " " + system_act["system_utterance"]

        ####################
        #   Photoshop Act  #
        ####################
        self.photoshop.observe(system_act)
        photoshop_act = self.photoshop.act()
        self.acts[3] = photoshop_act

        ############################
        #   Reset Upon Execution   #
        ############################
        print("sys_dialogue_act", sys_dialogue_act)
        if sys_dialogue_act == SystemAct.EXECUTE:
            self.system.reset()  # Reset state
            self.photoshop.reset()  # Clear everything

        # Return system text
        return sys_utt
