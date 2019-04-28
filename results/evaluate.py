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
    # mode: approve
    parser.add_argument('-b', '--batch-file', type=str,
                        help="AMT download batch file")
    parser.add_argument('-o', '--output-file', type=str,
                        help="Approve or reject file")

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
    elif mode == "approve":
        batch_file = args.batch_file
        output_file = args.output_file
        session_dir = args.dir
        approve(batch_file, args.dir, output_file)
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
        return sys_act['system_acts'][0]['dialogue_act']['value']

    sys_dialogue_acts = [get_dialogue_act(act[2]) for act in acts]

    total_edits = sys_dialogue_acts.count("execute")
    if 'total' not in obj:
        obj['total'] = {}
    if total_edits not in obj['total']:
        obj['total'][total_edits] = 0
    obj['total'][total_edits] += 1

    nedit = 1
    nturn = 0
    for dialogue_act in sys_dialogue_acts:
        nturn += 1
        if dialogue_act == "execute":
            if nedit not in obj:
                obj[nedit] = {}
            if nturn not in obj[nedit]:
                obj[nedit][nturn] = 0
            obj[nedit][nturn] += 1

            if nturn == 1:
                import pdb
                pdb.set_trace()

            nturn = 0
            nedit += 1

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

    print("[edits]")
    for metric, values in num_edits.items():
        print("metric", metric)
        print(json.dumps(values, sort_keys=True,
                         indent=4, separators=(',', ': ')))


def approve(batch_file, session_dir, output_file):
    """ Approve
    """

    outputs = []

    df = pd.read_csv(batch_file)

    seen_workers = set()

    print("df", df.shape)
    approved = 0
    rejected = 0

    for i, row in df.iterrows():
        print("dialogue", i)
        worker_id = df.loc[i, "WorkerId"]
        status = df.loc[i, "AssignmentStatus"]
        session_id = df.loc[i, "Answer.survey code"]

        if worker_id in seen_workers:
            reason = "You submitted more than one HIT.  Your additional HITs will be rejected."
            df.loc[i, "Reject"] = reason
        elif status == "Submitted" and pd.isnull(df.loc[i, "Reject"]):
            # We need to make a decision
            session_pickle = os.path.join(
                session_dir, 'session.{}.pickle'.format(session_id))

            if not os.path.exists(session_pickle):
                # Invalid survey code
                reason = "Your survey code is invalid. Click on the red button and the survey code will appear on the right. Afterwards, input the survey code into AMT platform"
                df.loc[i, "Reject"] = reason
            else:
                session = load_from_pickle(session_pickle)
                print_dialogue(session)

                decision = None
                while decision not in ["yes", "no"]:
                    decision = input("Approve? (yes/no): ")

                if decision == "yes":
                    df.loc[i, "Approve"] = "x"

                else:
                    df.loc[i, "Reject"] = "Our manual inspection shows that you did not follow the instructions."

        if isinstance(df.loc[i, "Reject"], str):
            rejected += 1
        else:
            approved += 1

        seen_workers.add(worker_id)
        outputs.append(row)

    print("Approved", approved)
    print("Rejected", rejected)

    df.to_csv(output_file)


if __name__ == "__main__":
    main()
