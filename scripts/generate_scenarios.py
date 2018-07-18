import argparse
import random
import pickle


from iedsp.ontology import Ontology

random.seed(3337)


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
                        default='scenario.pickle', help='Pickle file path to save tasks')
    args = parser.parse_args()

    util.save_to_pickle(scenarios, args.save)
