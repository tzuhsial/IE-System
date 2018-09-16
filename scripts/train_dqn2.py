from configparser import ConfigParser
import json
import os
import random
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

import numpy as np
from tqdm import tqdm

from cie import ImageEditEnvironment, EvaluationManager, util


def run_agendas(agendas,
                env,
                train_mode=False,
                train_config=None,
                global_step=None):

    print("train_mode", train_mode)

    user = env.agents[0]
    policy = env.agents[2].policy

    # Train
    if train_mode:
        random.shuffle(agendas)

    returns = []
    turns = []
    losses = []
    goals = []

    for agenda in tqdm(agendas):

        # setup agenda for new dialogue session

        R = 0
        reward_list = []
        turn = 0
        loss = 0
        done = False

        state = env.reset(agenda)

        while not done:  # While user has not terminated episode

            if train_mode:
                policy.update_epsilon(global_step)
            else:
                policy.update_epsilon(test=True)

            action = policy.step(state)

            next_state, reward, done, _ = env.step(action)

            if train_mode:
                tup = (state, action, reward, next_state, done)
                policy.replaymemory.add(*tup)
                batch_loss = policy.update_network()

                if global_step % train_config["freeze_interval"] == 0:
                    policy.copy_qnetwork()

                global_step += 1
            else:
                batch_loss = 0.

            state = next_state

            # Evaluation Manager
            turn += 1
            R += reward
            reward_list.append(reward)
            loss += batch_loss

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

    # Setup env & agents & policy
    world_config = config["world"]
    agents_config = config["agents"]
    env = ImageEditEnvironment(world_config, agents_config)

    # Load policy if specified
    policy = env.agents[2].policy
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
            train_summary = run_agendas(train_agendas, env, True, train_config,
                                        global_step)
            scribe.add_summary(epoch, 'train', train_summary)

            print("train")
            scribe.pprint_summary(train_summary)

            test_summary = run_agendas(
                test_agendas, env, global_step=global_step)
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
