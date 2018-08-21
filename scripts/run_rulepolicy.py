from configparser import ConfigParser
import logging
import json
import os
import random
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

import numpy as np
from tqdm import tqdm

from iedsp import ChannelPortal, UserPortal, SystemPortal, PhotoshopPortal, ImageEditWorld
from iedsp.policy import ActionMapper, DQNPolicy
from iedsp.evaluate import EvaluationManager
from iedsp import util


def print_mean_std(name, seq):
    seq_mean = np.mean(seq)
    seq_std = np.std(seq)
    print(name, "{}(+/-{})".format(seq_mean, seq_std))


def run_agendas(agendas, world):
    user = world.agents[0]
    Turns = []
    Returns = []
    Goals = []
    for agenda in tqdm(agendas):

        world.reset()
        user.load_agenda(agenda)

        episode_done = False
        turn = 0
        R = 0.

        while not episode_done:

            world.parley()

            reward = world.reward()  # User reward
            episode_done = world.episode_done()  # User episode_done

            # Evaluation Manager
            turn += 1
            R += reward

            # Finally
            if episode_done:
                break

        ngoals = user.completed_goals()

        Turns.append(turn)
        Returns.append(R)
        Goals.append(ngoals)

    return Turns, Returns, Goals


def main(argv):
    # Get config
    config_file = argv[1]
    config = util.load_from_json(config_file)

    # Setup world & agents & policy
    agents_config = config["agents"]
    user = UserPortal(agents_config["user"])
    channel = ChannelPortal(agents_config["channel"])
    system = SystemPortal(agents_config["system"])
    photoshop = PhotoshopPortal(agents_config["photoshop"])

    agents = [user, channel, system, photoshop]
    world_config = config["world"]
    world = ImageEditWorld(world_config, agents)

    # Load agendas

    train_agendas = util.load_from_pickle(config["agendas"]["train"])
    test_agendas = util.load_from_pickle(config["agendas"]["test"])

    # Main loop here
    """
    turns, returns, goals = run_agendas(train_agendas, world)
    print("Train")
    print_mean_std("turn", turns)
    print_mean_std("return", returns)
    print_mean_std("goals", goals)
    """
    turns, returns, goals = run_agendas(test_agendas, world)
    print("Test")
    print_mean_std("turn", turns)
    print_mean_std("return", returns)
    print_mean_std("goals", goals)


if __name__ == "__main__":
    main(sys.argv)
