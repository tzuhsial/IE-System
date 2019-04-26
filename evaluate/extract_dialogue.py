"""
Usage:
    python extract_dialogue.py SESSION_PICKLE
"""
import json
import os
import sys

import cie.util as util


def main():
    SESSION_PICKLE = sys.argv[1]
    session = util.load_from_pickle(sys.argv[1])


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

    print(json.dumps(session['survey'], sort_keys=True, indent=4, separators=(',', ': ')))
    print("[overall] edits", num_edits, "turns", num_turns)


if __name__ == "__main__":
    main()
