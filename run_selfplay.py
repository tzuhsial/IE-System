import itertools
import json
import random

from agent import Agent
from photoshopapi import PhotoshopAPI
from user import AgendaBasedUserSimulator as User
import util
from world import ImageEditWorld as World


def main():

    task_pickle = 'task.debug.pickle'

    tasks = util.load_from_pickle(task_pickle)
    profiles, goals = tasks

    photoshop = PhotoshopAPI()
    user = User()
    agent = Agent()
    world = World([photoshop, user, agent])

    for profile, goal in itertools.product(profiles, goals):

        user.set_profile(profile)
        user.set_goal(goal)

        world.reset()
        episode_done = False

        while not episode_done:

            world.parley()

            episode_done = world.episode_done()
            import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
