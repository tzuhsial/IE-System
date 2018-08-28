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


def run_agendas(agendas,
                world,
                train_mode=False,
                train_config=None,
                global_step=None):

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
        reward_list = []
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
            reward_list.append(reward)
            loss += batch_loss

            # Finally
            if episode_done:
                break

        # Freeze interval
        ngoal = user.completed_goals()

        returns.append(R)
        turns.append(turn)
        losses.append(loss)
        goals.append(ngoal)

    summary = {"return": returns, 'turn': turns, 'loss': losses, 'goal': goals}

    mode_prefix = "train/" if train_mode else "test/"
    policy.log_scalar(mode_prefix + "return", np.mean(returns), global_step)
    policy.log_scalar(mode_prefix + "turns", np.mean(turns), global_step)
    policy.log_scalar(mode_prefix + "losses", np.mean(losses), global_step)
    policy.log_scalar(mode_prefix + "goals", np.mean(goals), global_step)
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
    ignore_config = config["agents"]["system"]["actionmapper"]
    action_mapper = ActionMapper(ontology_json, ignore_config)
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
        for epoch in tqdm(range(1, train_config["num_epochs"] + 1, 1)):
            print("epoch", epoch)
            # Train
            train_summary = run_agendas(train_agendas, world, True,
                                        train_config, global_step)
            scribe.add_summary(epoch, 'train', train_summary)

            print("train")
            scribe.pprint_summary(train_summary)

            test_summary = run_agendas(
                test_agendas, world, global_step=global_step)
            scribe.add_summary(epoch, 'test', test_summary)

            print("train")
            scribe.pprint_summary(test_summary)
    except KeyboardInterrupt:
        print("Killed by hand")

    exp_path = train_config["save"]
    policy.save(exp_path)
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    history_path = os.path.join(exp_path, 'history.pickle')
    scribe.save(history_path)
    meta_path = os.path.join(exp_path, 'meta.json')
    util.save_to_json(config, meta_path)


if __name__ == "__main__":
    main(sys.argv)
