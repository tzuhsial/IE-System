import argparse
import random
import pickle

import lightroom
import photoshop
import user
import util


random.seed(3337)


def generate_multi_goal_agenda_lightroom():
    """
        Generates a mix of adjust & non adjust actions
    """
    num_of_adjust_slots = random.randint(5, 15)
    num_of_non_adjust_slots = random.randint(1, 4)

    adjust_slots = []
    for _ in range(num_of_adjust_slots):
        adjust_slot = random.choice(lightroom.schema.adjust_slots)
        min_value = lightroom.schema.adjust_slots_range[adjust_slot]['min']
        max_value = lightroom.schema.adjust_slots_range[adjust_slot]['max']
        value = random.randint(min_value, max_value)
        adjust_slots.append(
            {'slot': adjust_slot, 'value': value, 'type': 'adjust'})

    non_adjust_slots = []
    for _ in range(num_of_non_adjust_slots):
        non_adjust_slot = random.choice(lightroom.schema.non_adjust_slots)
        options = lightroom.schema.non_adjust_slots_options[non_adjust_slot]
        constraint = random.choice(options)
        non_adjust_slots.append(
            {'slot': non_adjust_slot, 'value': constraint, 'type': 'non_adjust'})

    goal = adjust_slots + non_adjust_slots
    random.shuffle(goal)

    return goal


def generate_photoshop_agenda():
    """
        Action example: { type: "edit", slot: "adjust", value: 50 }
    """
    num_of_slots = 5
    agenda = list()
    for _ in range(num_of_slots):
        action = photoshop.schema.random_action_factory()
        agenda.append(action)
    return agenda


goal_factory_dict = {
    0: generate_multi_goal_agenda_lightroom,
    1: generate_photoshop_agenda,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', dest='type', type=int,
                        default=0, help='Integer: 0 for lightroom | 1 for photoshop')
    parser.add_argument('-num-of-profiles', dest='num_of_profiles',
                        type=int, default=5, help='Generate number of random profiles to sample from')
    parser.add_argument('-num-of-goals', dest='num_of_goals',
                        type=int, default=100, help='Generate number of random goals to sample from')
    parser.add_argument('-num-of-scenarios', dest='num_of_tasks',
                        type=int, default=None, help='Number of tasks to generate')
    parser.add_argument('-save', dest='save', type=str,
                        default='scenario.photoshop.pickle', help='Pickle file path to save tasks')
    args = parser.parse_args()

    # Get arguments
    profile_factory = user.generate_random_profile
    goal_factory = goal_factory_dict.get(args.type)
    num_of_profiles = args.num_of_profiles
    num_of_goals = args.num_of_goals
    num_of_tasks = args.num_of_profiles * \
        args.num_of_goals if args.num_of_tasks is None else args.num_of_tasks

    # Generate Profile and Goals
    profiles = [profile_factory() for _ in range(num_of_profiles)]
    goals = [goal_factory() for _ in range(num_of_goals)]

    # Sample from profile and goal to generate number of tasks
    scenarios = []
    for _ in range(num_of_tasks):
        sampled_profile = random.choice(profiles)
        sampled_goal = random.choice(goals)
        scenario = (sampled_profile, sampled_goal)
        scenarios.append(scenario)

    util.save_to_pickle(scenarios, args.save)
