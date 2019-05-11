import argparse
import collections
import csv
import json
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    # function
    subparsers = parser.add_subparsers(
        help='Arguments for specific functions.', dest='function')
    subparsers.required = True

    # function: print
    print_parser = subparsers.add_parser("print")
    print_parser.add_argument('-s', '--session-pickle', required=True,
                              help='Path to session pickle')

    # function: visualize
    vis_parser = subparsers.add_parser("visualize")
    vis_parser.add_argument('-a', '--approved', required=True,
                            help="List of approved session ids")
    vis_parser.add_argument('-s', '--session-dir', required=True,
                            help="Path to session pickle directory")
    vis_parser.add_argument('-o', '--output-dir', required=True,
                            help="Output directory to store visualizations")

    # function: turns
    turn_parser = subparsers.add_parser("turns")
    turn_parser.add_argument('-a', '--approved', required=True,
                             help="List of approved session ids")
    turn_parser.add_argument('-s', '--session-dir', required=True,
                             help="Path to session pickle directory")
    turn_parser.add_argument('-o', '--output-csv', required=True,
                             help="Path to output csv")

    # function: study
    # Save like, dislike, suggest as csv to output dir
    # upload csv to spreadsheets for human labeling
    study_parser = subparsers.add_parser("study")
    study_parser.add_argument(
        '-a', '--approved', required=True, help="Path to approved session ids")
    study_parser.add_argument('-s', '--session-dir',
                              required=True, help="Path to session directory")
    study_parser.add_argument('-o', '--output-dir', required=True,
                              type=str, help="Output directory save like/dislike/suggestion")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    function = args.function
    if function == "print":
        session_pickle = args.session_pickle
        session = load_from_pickle(session_pickle)
        print_dialogue(session)
    elif function == "visualize":
        approved_file = args.approved
        session_dir = args.session_dir
        output_dir = args.output_dir
        visualize(approved_file, session_dir, output_dir)
    elif function == "turns":
        approved_file = args.approved
        session_dir = args.session_dir
        output_csv = args.output_csv
        analyze_turns(approved_file, session_dir, output_csv)
    elif function == "study":
        approved_file = args.approved
        session_dir = args.session_dir
        output_dir = args.output_dir
        save_like_dislike_suggest(approved_file, session_dir, output_dir)


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


def visualize(approved_file, session_dir, output_dir):
    """ Visualize all dialogues and save to output directory
    """
    raise NotImplementedError("Do this later")


def get_dialogue_act(sys_act):
    if 'system_acts' not in sys_act:
        # initial turn
        return 'greeting'
    else:
        return sys_act['system_acts'][0]['dialogue_act']['value']


def analyze_turns(approved_file, session_dir, output_csv):

    session_ids = read_sessions(approved_file)

    # Output csv rows
    rows = []

    # Constants
    EXECUTE = 'execute'
    QUERY = 'query'

    for session_id in session_ids:
        session_pickle = os.path.join(
            session_dir, 'session.{}.pickle'.format(session_id))

        session = load_from_pickle(session_pickle)

        acts = session['acts']

        orig_sys_acts = [get_dialogue_act(act[2]) for act in acts]

        # Filter Query
        sys_acts = [sys_act for sys_act in orig_sys_acts if sys_act != QUERY]

        # Metrics here

        # Total number of turns in dialogue
        nturns = len(sys_acts)

        # Total number of edits in dialogue
        nedits = sys_acts.count(EXECUTE)

        # Total number of turns to complete 2 edits
        is_execute_acts = [sys_act == EXECUTE for sys_act in sys_acts]
        accum_execute_acts = np.add.accumulate(is_execute_acts).tolist()

        # 1 edit
        if 1 not in accum_execute_acts:
            nturn_1edit = None
        else:
            nturn_1edit = accum_execute_acts.index(1)

        # 2 edit
        if 2 not in accum_execute_acts:
            nturn_2edit = None
        else:
            nturn_2edit = accum_execute_acts.index(2)

        # Add to existing rows
        row = [session_id, nturns, nedits, nturn_1edit, nturn_2edit]
        rows.append(row)

    with open(output_csv, 'w') as fout:
        writer = csv.writer(fout, delimiter=',', quotechar="\"")
        header = ["session_id", "nturns", "nedits",
                  "nturns for 1 edit", "nturns for 2 edits"]
        writer.writerow(header)
        writer.writerows(rows)


def save_like_dislike_suggest(approved_file, session_dir, output_dir):

    session_ids = read_sessions(approved_file)

    freeform = {
        'like': [],
        'dislike': [],
        'suggestion': []
    }

    categories = sorted(freeform.keys())

    for session_id in session_ids:
        session_pickle = os.path.join(
            session_dir, 'session.{}.pickle'.format(session_id))
        session = load_from_pickle(session_pickle)

        for category in categories:
            text = session['survey'][category].strip()
            freeform[category].append(text)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category, values in freeform.items():
        # Sort values here
        values = sorted(values)
        output_csv = os.path.join(output_dir, '{}.csv'.format(category))
        with open(output_csv, 'w') as fout:
            for text in values:
                fout.write(text+'\n')

#################
#     Util      #
#################


def read_sessions(approved):
    session_ids = []
    with open(approved, 'r') as fin:
        for line in fin.readlines():
            session_id = line.strip()
            session_ids.append(session_id)
    return session_ids


def load_from_pickle(filepath):
    with open(filepath, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


if __name__ == "__main__":
    main()
