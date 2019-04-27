"""
Parse approved session from downloaded batch file
Usage:
    python parse_approved.py BATCH_FILE OUTPUT_FILE
"""
import argparse
import csv
import os


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

    approved_ids = []

    status_col = None
    session_col = None

    with open(batch_file, 'r') as fin:
        reader = csv.reader(fin, delimiter=",")
        for row in reader:
            if status_col is None:
                status_col = row.index("AssignmentStatus")
                session_col = row.index("Answer.survey code")
                continue

            status = row[status_col]
            session_id = row[session_col]

            if status == "Approved":
                approved_ids.append(session_id)

    with open(output_file, 'w') as fout:
        for session_id in approved_ids:
            fout.write(str(session_id) + "\n")


if __name__ == "__main__":
    main()
