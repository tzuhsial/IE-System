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

    # Load Policy
    policy_config = config["policy"]

    # network input_size & output_size
    policy_config["qnetwork"]["input_size"] = len(system.state.to_list())
    ontology_json = util.load_from_json(config["ontology"])
    action_mapper = ActionMapper(ontology_json)
    policy_config["qnetwork"]["output_size"] = action_mapper.size()
    policy = DQNPolicy(policy_config, action_mapper)

    system.load_policy(policy)

    # Load agendas

    train_agendas = util.load_from_pickle(config["agendas"]["train"])
    test_agendas = util.load_from_pickle(config["agendas"]["test"])

    # Main loop here
    train_config = config["policy"]
    #scribe = EvaluationManager()

    # First burn_in memory

    global_step = 0
    for epoch in tqdm(range(1, train_config["num_epochs"]+1, 1)):
        print("epoch", epoch)
        # Train
        random.shuffle(train_agendas)

        for agenda in train_agendas:

            world.reset()
            user.load_agenda(agenda)

            R = 0
            turn = 0
            loss = 0
            episode_done = False
            while not episode_done:

                policy.update_epsilon(global_step)

                world.parley()

                reward = world.reward()  # User reward
                episode_done = world.episode_done()  # User episode_done

                # Extract
                policy.record(reward, episode_done)

                batch_loss = policy.update_network()

                if global_step % train_config["freeze_interval"] == 0:
                    policy.copy_qnetwork()

                # Evaluation Manager
                turn += 1
                R += reward
                loss += batch_loss
                global_step += 1
                
                # Finally
                if episode_done:
                    break

            ngoals = user.num_remaining_goals()


if __name__ == "__main__":
    main(sys.argv)
