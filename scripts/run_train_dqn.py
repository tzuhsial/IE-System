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
from cie.policy import ActionMapper, DQNPolicy
from cie.evaluate import EvaluationManager
from cie import util


def print_mean_std(name, seq):
    seq_mean = np.mean(seq)
    seq_std = np.std(seq)
    print(name, "{}(+/-{})".format(seq_mean, seq_std))


def run_agendas(agendas, world, train_mode=False, train_config=None, global_step=None):

    print("train_mode", train_mode)

    user = world.agents[0]
    policy = world.agents[2].policy

    # Train
    if train_mode:
        random.shuffle(agendas)

    returns = []
    turns = []
    losses = []
    goals = []

    for agenda in tqdm(agendas):

        world.reset()
        user.load_agenda(agenda)

        R = 0
        turn = 0
        loss = 0
        episode_done = False
        while not episode_done:
            if train_mode:
                policy.update_epsilon(global_step)
            else:
                policy.update_epsilon(test=True)

            world.parley()

            reward = world.reward()  # User reward
            episode_done = world.episode_done()  # User episode_done

            # Extract
            if train_mode:
                policy.record(reward, episode_done)

                batch_loss = policy.update_network()

                if global_step % train_config["freeze_interval"] == 0:
                    policy.copy_qnetwork()

                global_step += 1
            else:
                batch_loss = 0

            # Evaluation Manager
            turn += 1
            R += reward
            loss += batch_loss

            # Finally
            if episode_done:
                break

        ngoal = user.completed_goals()

        returns.append(R)
        turns.append(turn)
        losses.append(loss)
        goals.append(ngoal)

        import pdb
        pdb.set_trace()

    summary = {
        "return": returns,
        'turn': turns,
        'loss': losses,
        'goal': goals
    }
    return summary


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

    print("state size", policy_config["qnetwork"]["input_size"])
    print("action size", policy_config["qnetwork"]["output_size"])

    if policy_config.get("load") is not None:
        policy.load(policy_config["load"])

    system.load_policy(policy)

    # Load agendas
    train_agendas = util.load_from_pickle(config["agendas"]["train"])
    test_agendas = util.load_from_pickle(config["agendas"]["test"])
    print("train", len(train_agendas))
    print("test", len(test_agendas))
    # Main loop here
    train_config = config["policy"]
    scribe = EvaluationManager()

    # First burn_in memory

    global_step = 0
    try:
        for epoch in tqdm(range(1, train_config["num_epochs"]+1, 1)):
            print("epoch", epoch)
            """
            # Train
            train_summary = run_agendas(
                train_agendas, world, True, train_config, global_step)
            scribe.add_summary(epoch, 'train', train_summary)

            print("train")
            scribe.pprint_summary(train_summary)
            """
            test_summary = run_agendas(test_agendas, world)
            scribe.add_summary(epoch, 'test', test_summary)

            print("train")
            scribe.pprint_summary(test_summary)
    except KeyboardInterrupt:
        print("Killed by hand")

    import pdb
    pdb.set_trace()
    policy.save("")


if __name__ == "__main__":
    main(sys.argv)
