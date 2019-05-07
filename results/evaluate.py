import argparse
import collections
import csv
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('mode', type=str, help="mode (print|stats)")
    # mode: print
    parser.add_argument('-s', '--session', type=str,
                        help='Path to session pickle')
    # mode: stats
    parser.add_argument('-a', '--approved', type=str,
                        default="./evaluate/approved.txt", help="List of approved session ids")
    parser.add_argument('-d', '--dir', type=str,
                        help="Path to session pickle directory")

    args = parser.parse_args()
    return args


def load_from_pickle(filepath):
    """Load from pickle
    """
    with open(filepath, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def visualize(session):
    """
    Visualize a session object
    """
    raise NotImplementedError


def main():
    args = parse_args()
    mode = args.mode
    if mode == "print":
        session_pickle = args.session
        session = load_from_pickle(session_pickle)
        print_dialogue(session)
    elif mode == "stats":
        approved_file = args.approved
        session_dir = args.dir
        print("approved", approved_file)
        print("session", session_dir)
        calculate_stats(approved_file, session_dir)
    else:
        raise ValueError("Unknown mode: {}".format(mode))


def print_dialogue(session):
    num_edits = 0
    num_turns = 0

    for acts in session['acts'][1:]:
        user_utterance = acts[0].get("user_utterance", "")
        vision_utterance = acts[0].get('visionengine_utterance', "")
        system_utterance = acts[2].get("system_utterance", "")
        sys_act = acts[2]
        num_edits += sys_act['system_acts'][0]['dialogue_act']['value'] == "execute"
        if user_utterance:
            print("User:", user_utterance)
            num_turns += 1
        else:
            print("Vision:", vision_utterance)
        print("System:", system_utterance)

    print('image_id', session['image_id'])
    print(json.dumps(session['survey'], sort_keys=True,
                     indent=4, separators=(',', ': ')))
    print("[overall] edits", num_edits, "turns", num_turns)


def read_sessions(approved):
    session_ids = []
    with open(approved, 'r') as fin:
        for line in fin.readlines():
            session_id = line.strip()
            session_ids.append(session_id)
    return session_ids


def calculate_edits(acts, obj):
    """
    Calculate edits related metrics
    """

    def get_dialogue_act(sys_act):
        if 'system_acts' not in sys_act:
            # initial turn
            return 'greeting'
        else:
            return sys_act['system_acts'][0]['dialogue_act']['value']

    sys_dialogue_acts = [get_dialogue_act(act[2]) for act in acts]

    total_edits = sys_dialogue_acts.count("execute")
    if 'total' not in obj:
        obj['total'] = {}
    if total_edits not in obj['total']:
        obj['total'][total_edits] = 0
    obj['total'][total_edits] += 1

    nedit = 0
    nturn = 0
    total_turn = 0
    if 'turns_for_two_edits' not in obj:
        obj['turns_for_two_edits'] = {}

    for dialogue_act in sys_dialogue_acts:
        nturn += 1
        total_turn += 1
        if dialogue_act == "execute":
            nedit += 1
            if nedit not in obj:
                obj[nedit] = {}
            if nturn not in obj[nedit]:
                obj[nedit][nturn] = 0
            obj[nedit][nturn] += 1

            if nedit == 2:
                if total_turn not in obj['turns_for_two_edits']:
                    obj['turns_for_two_edits'][total_turn] = 0
                obj['turns_for_two_edits'][total_turn] += 1

            nturn = 0

    return obj


def calculate_stats(approved, pickle_dir):

    session_ids = read_sessions(approved)

    counter = {}
    num_edits = {}

    for session_id in session_ids:
        session_pickle = os.path.join(
            pickle_dir, 'session.{}.pickle'.format(session_id))

        session = load_from_pickle(session_pickle)

        # Acts
        acts = session['acts']

        # Counter number of turns
        nturns = len(acts)
        if 'nturns' not in counter:
            counter['nturns'] = {}
        if nturns not in counter['nturns']:
            counter['nturns'][nturns] = 0
        counter['nturns'][nturns] += 1

        # Count edit intervals
        calculate_edits(acts, num_edits)

        # Create survey
        survey = session['survey']

        for metric, value in survey.items():
            if metric not in counter:
                counter[metric] = {}
            if value not in counter[metric]:
                counter[metric][value] = 0

            counter[metric][value] += 1

    for metric, values in counter.items():
        print("metric", metric)
        print(json.dumps(values, sort_keys=True,
                         indent=4, separators=(',', ': ')))

        if metric in ['like', 'dislike', 'suggestion']:
            with open(metric + ".csv", 'w') as fout:
                fout.write(metric + '\n')
                for key in sorted(counter[metric].keys()):
                    value = counter[metric][key]
                    for _ in range(value):
                        fout.write(key + '\n')

    print("[edits]")
    for metric, values in num_edits.items():
        print("metric", metric)
        print(json.dumps(values, sort_keys=True,
                         indent=4, separators=(',', ': ')))


if __name__ == "__main__":
    main()