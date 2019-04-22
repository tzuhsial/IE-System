"""
Usage:
    python extract_dialogue.py SESSION_PICKLE
"""
import os
import sys

import cie.util as util


def main():

    SESSION_PICKLE = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]

    session = util.load_from_pickle(sys.argv[1])

    for acts in session['acts'][1:]:
        user_utterance = acts[0].get("user_utterance", "")
        vision_utterance = acts[0].get('visionengine_utterance', "")
        system_utterance = acts[2].get("system_utterance", "")

        if user_utterance:
            print("User:", user_utterance)
        else:
            print("Vision:", vision_utterance)
        print("System:", system_utterance)


if __name__ == "__main__":
    main()
