import logging
import json
import os
import pprint
import random
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

from configparser import ConfigParser

import numpy as np
# from flask import Flask, jsonify, request

from cie import TrackerPortal, SystemPortal, PhotoshopPortal
from cie import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# app = Flask(__name__)

pp = pprint.PrettyPrinter(indent=2)


def main(argv):
    config_file = argv[1]
    config = util.load_from_json(config_file)

    # Load agents here
    agents_config = config["agents"]
    tracker = TrackerPortal(agents_config["tracker"])
    system = SystemPortal(agents_config["system"])
    photoshop = PhotoshopPortal(agents_config["photoshop"])

    tracker.reset()
    system.reset()
    photoshop.reset()

    photoshop_act = {}
    while True:
        sentence = input("User: ")

        user_act = {"user_utterance": sentence,
                    "episode_done": False, "reward": 0}

        tracker.observe(user_act)
        tracker_act = tracker.act()

        pp.pprint(tracker_act["user_acts"][0])

        system.observe(photoshop_act)
        system.observe(tracker_act)
        system_act = system.act()

        print("System:", system_act["system_utterance"])
        photoshop.observe(system.act())
        photoshop_act = photoshop.act()


if __name__ == "__main__":
    main(sys.argv)
