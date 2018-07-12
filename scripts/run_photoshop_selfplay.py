import itertools
import json
import random

from agent import Agent
from opt import parse_opt
from photoshop import PhotoshopAPI
from user import AgendaBasedUserSimulator as User
import util
from world import ImageEditWorld as World


def main():
    opt = parse_opt()
    scenarios = util.load_from_pickle(opt['scenario_pickle'])

    photoshop = PhotoshopAPI()
    user = User(opt)
    agent = Agent()
    world = World([photoshop, user, agent])

    for profile, agenda in scenarios:
        world.reset()
        user.setup_scenario(profile, agenda)
        user.print_agenda()
        episode_done = False
        import pdb
        pdb.set_trace()

        while not episode_done:
            world.parley()
            episode_done = world.episode_done()


if __name__ == "__main__":
    main()
