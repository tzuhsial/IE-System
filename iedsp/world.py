
class SelfPlayWorld(object):
    def __init__(self, agents):
        """
        Args:
            agents (list):  user 2. channel 3. agent 4. photoshop
        """
        self.agents = agents

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

        print("[User]", self.acts[user_idx]['user_utterance'], \
             'reward', self.acts[user_idx]['reward'])

        # Channel observes User
        channel.observe(self.acts[user_idx])
        self.acts[channel_idx] = channel.act()

        # Agent observes User from Channel and Photoshop
        system.observe(self.acts[channel_idx])
        system.observe(self.acts[photoshop_idx])
        self.acts[system_idx] = system.act()
        print("[System]", self.acts[system_idx]['system_utterance'])

        # Photoshop observes agent action
        photoshop.observe(self.acts[system_idx])
        self.acts[photoshop_idx] = photoshop.act()

        self.turn_count += 1
        print("[World] Turn", self.turn_count)

    def episode_done(self):
        """
        Return user episode done
        """
        return self.acts[0]['episode_done']
