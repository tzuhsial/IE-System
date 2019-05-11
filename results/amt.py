"""
Put all Amazon Mechinal Turk (AMT) related script here
"""
import argparse
import collections
import csv
import os
import sys
import pickle

import pandas as pd
from flask import Flask, send_from_directory
from nltk import word_tokenize


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help='Arguments for specific functions.', dest='function')
    subparsers.required = True

    # Approve: approved HITs from AMT
    approve_dialogue_parser = subparsers.add_parser("approve_dialogue")
    approve_dialogue_parser.add_argument(
        '-b', '--batch-file', required=True, help="Batch file downloaded from AMT results.")
    approve_dialogue_parser.add_argument(
        '-o', '--output-file', required=True, help="Approved batch file to upload to AMT.")
    approve_dialogue_parser.add_argument(
        '-s', '--session-dir', required=True, help="Path to pickled sessions directory")

    # Extract: extract approved session ids
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument(
        '-b', '--batch-file', required=True, help="Reviewed Batch file from AMT")
    extract_parser.add_argument(
        '-o', '--output-file', required=True, help="Output file of approved session ids.")

    # Prepare HIT: create HIT csv to upload to AMT
    prepare_hit_parser = subparsers.add_parser("prepare_hit")
    prepare_hit_parser.add_argument(
        '-a', '--approved', required=True, help="Path to approved session ids")
    prepare_hit_parser.add_argument(
        '-s', '--session-dir', required=True, help="Path to pickled sessions directory")
    prepare_hit_parser.add_argument(
        '-o', '--output-csv', required=True, help="Output csv file of prepared HIT for AMT")

    # Approve: approved NLU HITs from AMT
    approve_nlu_parser = subparsers.add_parser("approve_nlu")
    approve_nlu_parser.add_argument(
        '-b', '--batch-file', required=True, help="Batch file downloaded from AMT results.")
    approve_nlu_parser.add_argument(
        '-o', '--output-file', required=True, help="Approved batch file to upload to AMT.")

    # process nlu: process approved nlu HITs and outputs to file in ATIS format
    process_nlu_parser = subparsers.add_parser("process_nlu")
    process_nlu_parser.add_argument(
        '-b', '--batch-file', required=True, help="Batch file downloaded from AMT results.")
    process_nlu_parser.add_argument(
        '-o', '--output-file', required=True, help="Approved batch file to upload to AMT.")

    # Serve Image: serve images for HIT
    serve_image_parser = subparsers.add_parser("serve_image")
    serve_image_parser.add_argument(
        '-i', '--image-dir', required=True, help="Path to image directory")
    serve_image_parser.add_argument(
        '-p', '--port', default=8000, help="port number")
    serve_image_parser.add_argument(
        '-d', '--debug', action="store_true", help="Flask server debug mode")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.function == "approve_dialogue":
        approve_dialogue(args.batch_file, args.session_dir, args.output_file)
    elif args.function == "extract":
        extract(args.batch_file, args.output_file)
    elif args.function == "prepare_hit":
        prepare_hit(args.approved, args.session_dir, args.output_csv)
    elif args.function == "approve_nlu":
        approve_nlu(args.batch_file, args.output_file)
    elif args.function == "process_nlu":
        process_nlu(args.batch_file, args.output_file)
    elif args.function == "serve_image":
        serve_image(args.image_dir, args.port, args.debug)


def approve_dialogue(batch_file, session_dir, output_file):
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
        image_id = df.loc[i, "Input.image_id"]
        worker_id = df.loc[i, "WorkerId"]
        status = df.loc[i, "AssignmentStatus"]
        session_id = df.loc[i, "Answer.survey code"]
        print('session_id', session_id)
        if status != "Submitted":
            if worker_id in seen_workers:
                reason = "You submitted more than one HIT.  Your additional HITs will be rejected."
                df.loc[i, "Reject"] = reason
            elif status == "Submitted" and pd.isnull(df.loc[i, "Reject"]):
                # We need to make a decision
                session_pickle = os.path.join(
                    session_dir, 'session.{}.pickle'.format(session_id))

                if not os.path.exists(session_pickle):
                    # Invalid survey code
                    print("invalid session_id:", session_id)
                    reason = "Your survey code is invalid. Click on the red button and the survey code will appear on the right. Afterwards, input the survey code into AMT platform"
                    df.loc[i, "Reject"] = reason
                else:
                    df.loc[i, "Approve"] = "x"

            if status == "Approved" and isinstance(df.loc[i, "Approve"], str):
                approved += 1
            else:
                print("image_id", image_id, df.loc[i, "Reject"])
                rejected += 1
        else:
            print(row)
            import pdb
            pdb.set_trace()

        seen_workers.add(worker_id)
        outputs.append(row)

    print("Approved", approved)
    print("Rejected", rejected)

    df.to_csv(output_file)


def extract(batch_file, output_file):
    """ Extract approved session ids
    """
    df = pd.read_csv(batch_file)

    approved_ids = []
    for status in ["Approved", "Submitted"]:
        approved_ids += df[df['AssignmentStatus'] ==
                           status]['Answer.survey code'].tolist()

    with open(output_file, 'w') as fout:
        for session_id in approved_ids:
            fout.write(str(session_id) + '\n')


def prepare_hit(approved_file, session_dir, output_csv):
    """ Prepare HIT to upload to AMT
    Order:
    session_id, image_id, prev_system_utterance, turn_id, user_utterance, bounded_user_utterance
    """

    # Read session ids
    approved_ids = read_session_ids(approved_file)
    print("number of sessions", len(approved_ids))

    # Get CSV Rows
    hit_rows = []

    for session_id in approved_ids:
        print("session:", session_id)
        session_pickle = os.path.join(
            session_dir, 'session.{}.pickle'.format(session_id))
        session = load_from_pickle(session_pickle)

        image_id = session['image_id']

        # print("turn", 0)
        prev_system_utterance = "Hi! This is an image editing chatbot. How may I help you?"
        for turn_id, acts in enumerate(session['acts'][1:], 1):
            # print('system:', prev_system_utterance)
            # print('turn', turn_id)

            user_utterance = acts[0].get("user_utterance", "")
            # print("user:", user_utterance)

            # print("system:", system_utterance)

            if user_utterance:
                words = word_tokenize(user_utterance)
                user_utterance = " ".join(words)

                bounded = []
                for word_id, word in enumerate(words):
                    closure = "|{}|".format(word_id)
                    bounded.append(closure)
                    bounded.append(word)

                final_closure = '|{}|'.format(len(words))
                bounded.append(final_closure)

                bounded_user_utterance = ' '.join(bounded)
                # print("bounded:", bounded_user_utterance)
                # Order

                row = [session_id, image_id, prev_system_utterance,
                       turn_id, user_utterance, bounded_user_utterance]

                hit_rows.append(row)

            prev_system_utterance = acts[2].get('system_utterance', "")
        # print()

    # Write to CSV
    with open(output_csv, 'w') as fout:
        writer = csv.writer(fout, delimiter=",", quotechar="\"")
        header = ["session_id", "image_id", "prev_system_utterance",
                  "turn_id", "user_utterance", "bounded_user_utterance"]
        writer.writerow(header)
        for row in hit_rows:
            writer.writerow(row)


def approve_nlu(batch_file, output_file):

    df = pd.read_csv(batch_file)
    for i, row in df.iterrows():

        reject = ""

        prev_sys_utt = df.loc[i, "Input.prev_system_utterance"]
        usr_utt = df.loc[i, "Input.user_utterance"]

        words = usr_utt.split()

        slots = ["action", "refer", "attribute", "value"]
        pos = {}
        for slot in slots:
            pos[slot] = {
                'start': df.loc[i, "Answer.{}-start".format(slot)],
                'end': df.loc[i, "Answer.{}-end".format(slot)]
            }

        print("System:", prev_sys_utt)
        print("User:", usr_utt)

        for slot in slots:
            start = pos[slot]['start']
            end = pos[slot]['end']

            if end < start:
                reject += "Your {} end index is smaller than start index."\
                    .format(slot)

            nwords = end-start
            if slot in ["action", "attribute", "value"] and nwords > 1:
                reject += "Please select a one word span for {}. ".format(slot)

            slot_value = " ".join(words[start:end])
            if slot == "attribute" and slot_value not in ["brightness", "contrast", "hue", "saturation", "lightness"]:
                reject += "Please select one of \"brightness\", \"contrast\", \"hue\", \"saturation\", \"lightness\" for attribute."

            print("{}: {}".format(slot, slot_value))

        da = None
        if df.loc[i, "Answer.affirm.on"]:
            da = "affirm"
        elif df.loc[i, "Answer.negate.on"]:
            da = "negate"
        elif df.loc[i, "Answer.other.on"]:
            da = "inform"
        print("da:", da)

        if reject != "":
            df.loc[i, "Approve"] = "x"
        else:
            df.loc[i, "Reject"] = reject

        df.to_csv(output_file)


def preprocess(tokens):
    """ Preprocess tokens to tag file
    """
    words = []
    for token in tokens:
        if token in ["brightness", "contrast", "hue", "saturation", "lightness"]:
            word = "<attribute>"
        elif token.isdigit() or (token[0] in ["-", "+"] and token[1:].isdigit()):
            word = "<value>"
        else:
            word = token
        words.append(word)
    return words


def process_nlu(batch_file, output_file):

    df = pd.read_csv(batch_file)

    # Filter only approved
    df = df[df['Approve'] == 'x']
    print("number of annotations", len(df))

    session_ids = df['Input.session_id'].unique().tolist()

    outputs = []

    # utterances
    for session_id in session_ids:
        # Get all rows for session_id
        session_df = df[df["Input.session_id"] == session_id]

        # Get all turn_ids
        turn_ids = session_df['Input.turn_id'].unique().tolist()

        for turn_id in turn_ids:

            # Annotations
            ann = session_df[session_df['Input.turn_id'] == turn_id]

            # Get tokens
            user_utterance = ann['Input.user_utterance'].tolist()[0]

            tokens = user_utterance.lower().split()

            tokens = preprocess(tokens)

            # Get tags
            tags = ['O'] * len(tokens)

            slots = ["action", "refer", "attribute", "value"]

            for slot in slots:

                slot_counter = collections.Counter()
                for i, row in ann.iterrows():
                    start = row['Answer.{}-start'.format(slot)]
                    end = row['Answer.{}-end'.format(slot)]

                    pair = (start, end)
                    slot_counter[pair] += 1

                most_common_pair, freq = slot_counter.most_common()[0]

                mc_start, mc_end = most_common_pair

                if mc_start >= mc_end:
                    continue

                for i, pos in enumerate(range(mc_start, mc_end)):
                    if i == 0:
                        B_OR_I = 'B'
                    else:
                        B_OR_I = 'I'

                    tag = '{}-{}'.format(B_OR_I, slot)

                    tags[pos] = tag

            # Multiple annotations
            da_counter = collections.Counter()
            das = ["affirm", "negate", "other"]
            for da_name in das:
                da_counter[da_name] += \
                    ann['Answer.{}.on'.format(da_name)].sum()

            # Get dialogue_act
            da, freq = da_counter.most_common()[0]
            if da == "other":
                da = "inform"

            # 3 tuple
            # (tokens, tags, dialogue_act)

            assert len(tokens) == len(tags)

            combined = ['###'.join(x) for x in zip(tokens, tags)]
            tagged_tokens = ' '.join(combined)

            tup = (tagged_tokens, da)

            outputs.append(tup)

    with open(output_file, 'w') as fout:
        writer = csv.writer(fout, delimiter='|')
        header = ['tags', 'dialogue_act']
        writer.writerow(header)
        writer.writerows(outputs)


def serve_image(image_dir, port, debug):
    """ minimalist implementation of server and opts 
    """
    print('image_dir', image_dir)
    print("port", port)
    print("debug", debug)
    app = Flask(__name__)

    @app.route("/images/<image_id>", methods=["GET"])
    def send_image(image_id):
        image_name = image_id + ".jpg"
        return send_from_directory(image_dir, image_name)

    app.run(host="0.0.0.0", port=port, debug=debug)

################
#     Util     #
################


def read_session_ids(approved_file):
    session_ids = []
    with open(approved_file, 'r') as fin:
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
