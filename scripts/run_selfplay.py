from configparser import ConfigParser
import logging
import json
import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

import numpy as np
from tqdm import tqdm

from iedsp import util
from iedsp.channel import ChannelPortal
from iedsp.photoshop import PhotoshopPortal
from iedsp.system import System
from iedsp.user import UserPortal
from iedsp.world import SelfPlayWorld


def print_mean_std(name, seq):
    seq_mean = np.mean(seq)
    seq_std = np.std(seq)
    print(name, "{}(+/-{})".format(seq_mean, seq_std))


def main(argv):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    # Get arguments
    config_file = argv[1]

    # Configsc
    config = ConfigParser(allow_no_value=True)
    config.read(config_file)

    agendas = util.load_from_pickle(config['DEFAULT']['AGENDA_PICKLE'])

    # Build Interaction World
    user = UserPortal(config)
    channel = ChannelPortal(config)
    system = System(config)  # system needs global_config
    photoshop = PhotoshopPortal(config['PHOTOSHOP'])

    agents = [user, channel, system, photoshop]

    world = SelfPlayWorld(agents)

    turn_counts = []
    episode_rewards = []
    completed_goals = []
    for agenda in tqdm(agendas):
        world.reset()
        user.load_agenda(agenda)

        while True:

            world.parley()

            if world.episode_done():
                break

        turn_counts.append(world.turn_count)
        episode_rewards.append(system.get_reward())
        completed_goals.append(7 - len(user.agenda))

    print_mean_std('turn', turn_counts)
    print_mean_std('reward', episode_rewards)
    print_mean_std('goal', completed_goals)


if __name__ == "__main__":
    main(sys.argv)
