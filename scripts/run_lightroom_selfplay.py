import itertools
import json
import random

from agent import Agent
from lightroom import LightroomAPI
from user import AgendaBasedUserSimulator as User
import util
from world import ImageEditWorld as World


def main():

    scenario_pickle = 'scenario.lightroom.pickle'

    scenarios = util.load_from_pickle(scenario_pickle)

    lightroom = LightroomAPI()
    user = User()
    agent = Agent()
    world = World([lightroom, user, agent])

    for profile, multi_goal in scenarios:
        world.reset()
        user.setup_scenario(profile, multi_goal)
        user.print_agenda()
        episode_done = False

        while not episode_done:

            world.parley()

            episode_done = world.episode_done()

        import pdb
        pdb.set_trace()


if __name__ == "__main__":
    main()
