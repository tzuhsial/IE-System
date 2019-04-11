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

from cie import ChannelPortal, UserPortal, SystemPortal, PhotoshopPortal, ImageEditWorld
from cie.evaluate import EvaluationManager
from cie import util


def print_mean_std(name, seq):
    seq_mean = np.mean(seq)
    seq_std = np.std(seq)
    print(name, "{}(+/-{})".format(seq_mean, seq_std))


def run_agendas(agendas, world):
    user = world.agents[0]
    policy = world.agents[2].policy
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

            policy.record(reward, episode_done)

            # Evaluation Manager
            turn += 1
            R += reward

            # Finally
            if episode_done:
                break

        #import pdb
        # pdb.set_trace()

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
    world_config = config["world"]
    agents_config = config["agents"]
    world = ImageEditWorld(world_config, agents_config)

    # Load agendas
    train_agendas = util.load_from_pickle(config["agendas"]["train"])
    test_agendas = util.load_from_pickle(config["agendas"]["test"])
    print("train", len(train_agendas), "test", len(test_agendas))

    # Main loop here
    policy_config = config["agents"]["system"]["policy"]
    """
    if policy_config["save_replaymemory"]:
        turns, returns, goals = run_agendas(train_agendas, world)
        print("Train")
        print_mean_std("turn", turns)
        print_mean_std("return", returns)
        print_mean_std("goals", goals)
        print("success rate", (np.array(goals) == 3).mean())

        experience = world.agents[2].policy.replaymemory.storage
        util.save_to_pickle(experience, policy_config['save_replaymemory'])
    """
    turns, returns, goals = run_agendas(test_agendas, world)
    print("Test")
    print_mean_std("turn", turns)
    print_mean_std("return", returns)
    print_mean_std("goals", goals)

    print("success rate", (np.array(goals) == 3).mean())


if __name__ == "__main__":
    main(sys.argv)
