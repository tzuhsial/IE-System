"""
    The world where the image edit interaction takes place
"""

class ImageEditWorld(object):
    def __init__(self, agents, verbose=True):
        """
            Arguments:
            - agents: 1. apiapi 2. user 3. agent
        """
        self.agents = agents

        self.verbose = verbose

    def reset(self):

        for agent in self.agents:
            agent.reset()

        self.acts = [{}] * len(self.agents)

        self.turn_count = 0

    def parley(self):
        """
            The image observation is provided by api
            api observes agent
            User observes api and agent
            Agent observes api and user
        """
        # 0, 1, 2
        api, user, agent = self.agents

        # First api
        self.acts[0] = api.act()
        api.observe(self.acts[2])

        # Then User
        user.observe(self.acts[0])
        user.observe(self.acts[2])
        self.acts[1] = user.act()

        # Finally Agent
        agent.observe(self.acts[0])
        agent.observe(self.acts[1])
        self.acts[2] = agent.act()

        # api then observes the agent act
        api.observe(self.acts[2])

        if self.verbose:
            print(self.agents[0].name, self.acts[0])
            print(self.agents[1].name, self.acts[1]['user_acts'])
            print(self.agents[2].name, self.acts[2]['system_acts'])

        self.turn_count += 1

    def episode_done(self):
        return self.acts[1]['episode_done'] or self.acts[2]['episode_done']
