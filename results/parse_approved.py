"""
Parse approved session from downloaded batch file
Usage:
    python parse_approved.py BATCH_FILE OUTPUT_FILE
"""
import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_file', type=str,
                        help="Downloaded batch file from Mechanical Turk")
    parser.add_argument('-o', '--output_file', type=str,
                        help="Filepath to save approved session ids")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    batch_file = args.batch_file
    output_file = args.output_file

    df = pd.read_csv(batch_file)

    approved_ids = []
    for status in ["Approved", "Submitted"]:
        approved_ids += df[df['AssignmentStatus'] == status]['Answer.survey code'].tolist()

    with open(output_file, 'w') as fout:
        for session_id in approved_ids:
            fout.write(str(session_id) + '\n')


if __name__ == "__main__":
    main()
