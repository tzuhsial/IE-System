import argparse
import random
import pickle

import photoshopapi
import user
import util


def generate_profiles(num_of_profiles):
    list_of_profiles = list()

    for _ in range(num_of_profiles):
        profile = user.Profile.generate_random_profile()
        list_of_profiles.append(profile)

    return list_of_profiles


def generate_goals(num_of_tasks):

    list_of_goals = list()

    for _ in range(num_of_tasks):

        goal = {}
        for slot in photoshopapi.adjust_slots:
            # Randomly generate an integer between -100 and 100
            value = random.randint(-100, 100)
            goal[slot] = value
        list_of_goals.append(goal)

    return list_of_goals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-num-of-profiles', dest='num_of_profiles',
                        type=int, default=5, help='Number of user profiles to generate')
    parser.add_argument('-num-of-goals', dest='num_of_goals',
                        type=int, default=100, help='Number of goals to generate')
    parser.add_argument('-save', dest='save', type=str,
                        default='task.debug.pickle', help='Pickle file path to save tasks')
    args = parser.parse_args()

    profiles = generate_profiles(args.num_of_profiles)
    goals = generate_goals(args.num_of_goals)

    obj = (profiles, goals)
    util.save_to_pickle(obj, args.save)
