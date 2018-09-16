from .channel import ChannelPortal
from .user import UserPortal
from .system import SystemPortal
from .photoshop import PhotoshopPortal


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
