from .channel import ChannelPortal
from .user import UserPortal
from .system import SystemPortal
from .photoshop import PhotoshopPortal


class ImageEditWorld(object):
    def __init__(self, config, agents_config):
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
