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
                update_steps={}):

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

        R = 0
        loss = 0
        done = False

        state = env.reset(agenda)

        opt2act = policy.opt2act
        meta_policy = policy.meta_intent_policy
        if train_mode:
            meta_train_config = train_config["meta_intent_policy"]
            intent_train_config = train_config["intent_policy"]

        while not done:  # Dialogue Session

            # Meta controller decides a sub policy
            if train_mode:
                meta_step = update_steps["meta"]
                meta_policy.update_epsilon(meta_step)
            else:
                meta_policy.update_epsilon(test=True)

            option_idx = meta_policy.step(state)

            if option_idx in policy.primitive_actions:
                action = opt2act[option_idx]
                next_state, reward, done, _ = env.step(action)
                R += reward

            else:
                intent_name = policy.action_mapper\
                            .config["execute"][option_idx -2]
                option_done = False
                intent_execution_idx = policy.action_mapper.find_action_idx(
                    'execute', intent=intent_name)
                intent_policy = policy.intent_policies[option_idx]

                reward = 0

                while not option_done:
                    if train_mode:
                        intent_step = update_steps[option_idx]
                        intent_policy.update_epsilon(intent_step)
                    else:
                        intent_policy.update_epsilon(test=True)

                    # Select an action using the action & sub policy
                    intent_action = intent_policy.step(state)
                    action = opt2act[option_idx][intent_action]
                    next_state, intent_reward, done, _ = env.step(action)
                    R += reward

                    reward += intent_reward

                    #######################
                    #   Internal Critic   #
                    #######################
                    dialogue_state = env.agents[2].state
                    intent_prob = dialogue_state.get_slot("intent")\
                                .value_conf_map.get(intent_name, 0.0)

                    stochastic_termination = random.random() > intent_prob

                    intent_executed = (intent_execution_idx == action)
                    execute_success = dialogue_state.get_slot(
                        'execute_result').get_max_value()

                    option_success = intent_executed and execute_success
                    option_done = done or option_success or stochastic_termination

                    # Now we update our controller policy
                    if train_mode:
                        if option_success:
                            ic_reward = train_config['internal_critic'][
                                "option_success_reward"]
                            option_done = True
                        else:
                            ic_reward = train_config["internal_critic"][
                                "turn_penalty"]
                        tup = (state, intent_action, ic_reward, next_state,
                               option_done)
                        intent_policy.replaymemory.add(*tup)
                        batch_loss = intent_policy.update_network()
                        update_steps[option_idx] += 1

                        if update_steps[option_idx] % intent_train_config["freeze_interval"] == 0:
                            intent_policy.copy_qnetwork()

                    update_steps[option_idx] += 1

                    state = next_state

            # Update meta policy here
            if train_mode:
                tup = (state, option_idx, reward, next_state, done)
                meta_policy.replaymemory.add(*tup)
                batch_loss = meta_policy.update_network()

                if update_steps["meta"] % meta_train_config["freeze_interval"] == 0:
                    meta_policy.copy_qnetwork()

                update_steps["meta"] += 1
            else:
                batch_loss = 0.

            state = next_state

            # Evaluation Manager
            loss += batch_loss

        ngoal = user.completed_goals()

        returns.append(R)
        turns.append(env.turn_count)
        losses.append(loss)
        goals.append(ngoal)

    summary = {"return": returns, 'turn': turns, 'loss': losses, 'goal': goals}
    """
    mode_prefix = "train/" if train_mode else "test/"
    policy.log_scalar(mode_prefix + "return", np.mean(returns),
                      update_steps["meta"])
    policy.log_scalar(mode_prefix + "turns", np.mean(turns),
                      update_steps["meta"])
    policy.log_scalar(mode_prefix + "losses", np.mean(losses),
                      update_steps["meta"])
    policy.log_scalar(mode_prefix + "goals", np.mean(goals),
                      update_steps["meta"])
    """
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
    # Initialize update steps for all policies
    update_steps = {}
    update_steps['meta'] = 0
    for option_idx in policy.intent_policies.keys():
        update_steps[option_idx] = 0

    try:
        for epoch in tqdm(range(1, train_config["num_epochs"] + 1, 1)):
            print("epoch", epoch)
            # Train
            train_summary = run_agendas(
                train_agendas,
                env,
                True,
                train_config,
                update_steps=update_steps)
            scribe.add_summary(epoch, 'train', train_summary)

            print("train")
            scribe.pprint_summary(train_summary)

            test_summary = run_agendas(
                test_agendas, env, update_steps=update_steps)
            scribe.add_summary(epoch, 'test', test_summary)

            print("test")
            scribe.pprint_summary(test_summary)
    except KeyboardInterrupt:
        print("Killed by hand")
    """
    exp_path = train_config["save"]
    policy.save(exp_path)
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    history_path = os.path.join(exp_path, 'history.pickle')
    print("Saving history to {}".format(history_path))
    scribe.save(history_path)
    meta_path = os.path.join(exp_path, 'meta.json')
    util.save_to_json(config, meta_path)
    """


if __name__ == "__main__":
    main(sys.argv)
