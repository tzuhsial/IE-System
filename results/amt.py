"""
Put all Amazon Mechinal Turk (AMT) related script here
"""
import argparse
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
    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument(
        '-b', '--batch-file', required=True, help="Batch file downloaded from AMT results.")
    approve_parser.add_argument(
        '-o', '--output-file', required=True, help="Approved batch file to upload to AMT.")
    approve_parser.add_argument(
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

    if args.function == "approve":
        approve(args.batch_file, args.session_dir, args.output_file)
    elif args.function == "extract":
        extract(args.batch_file, args.output_file)
    elif args.function == "prepare_hit":
        prepare_hit(args.approved, args.session_dir, args.output_csv)
    elif args.function == "serve_image":
        serve_image(args.image_dir, args.port, args.debug)


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

        #print("turn", 0)
        prev_system_utterance = "Hi! This is an image editing chatbot. How may I help you?"
        for turn_id, acts in enumerate(session['acts'][1:], 1):
            #print('system:', prev_system_utterance)
            #print('turn', turn_id)

            user_utterance = acts[0].get("user_utterance", "")
            #print("user:", user_utterance)

            #print("system:", system_utterance)

            if user_utterance:
                words = word_tokenize(user_utterance)

                bounded = []
                for word_id, word in enumerate(words):
                    closure = "|{}|".format(word_id)
                    bounded.append(closure)
                    bounded.append(word)

                final_closure = '|{}|'.format(len(words))
                bounded.append(final_closure)

                bounded_user_utterance = ' '.join(bounded)
                #print("bounded:", bounded_user_utterance)
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
