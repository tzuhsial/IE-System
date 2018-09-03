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
        turn = 0
        loss = 0
        episode_done = False

        while not episode_done:
            world.parley()

            reward = world.reward()
            episode_done = world.episode_done()

            # Extract
            policy.record(reward, episode_done)

            # Evaluation Manager
            turn += 1
            R += reward

            # Finally
            if episode_done:
                loss = policy.end_episode(train_mode)
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
    actor_losses, critic_losses = zip(*losses)
    policy.log_scalar(mode_prefix + "actor_losses", np.mean(actor_losses),
                      global_step)
    policy.log_scalar(mode_prefix + "critic_losses", np.mean(critic_losses),
                      global_step)
    policy.log_scalar(mode_prefix + "goals", np.mean(goals), global_step)
    return summary


def main(argv):
    # Get config
    config_file = argv[1]
    config = util.load_from_json(config_file)

    # Setup world & agents & policy
    world_config = config["world"]
    agents_config = config["agents"]
    world = ImageEditWorld(world_config, agents_config)

    # Load policy if specified
    policy = world.agents[2].policy
    policy_config = config["agents"]["system"]["policy"]
    if policy_config.get("load") is not None:
        policy.load(policy_config["load"])

    # Load agendas
    train_agendas = util.load_from_pickle(config["agendas"]["train"])
    test_agendas = util.load_from_pickle(config["agendas"]["test"])
    print("train", len(train_agendas))
    print("test", len(test_agendas))
    # Main loop here
    train_config = policy_config
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

            print("test")
            scribe.pprint_summary(test_summary)
    except KeyboardInterrupt:
        print("Killed by hand")

    exp_path = train_config["save"]
    policy.save(exp_path)
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    history_path = os.path.join(exp_path, 'history.pickle')
    print("Saving history to {}".format(history_path))
    scribe.save(history_path)
    meta_path = os.path.join(exp_path, 'meta.json')
    util.save_to_json(config, meta_path)


if __name__ == "__main__":
    main(sys.argv)
