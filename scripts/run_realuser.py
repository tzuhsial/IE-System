import argparse
import json
import logging
import os
import pprint
import random
import sys
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)


from cie import PhotoshopPortal, SystemPortal, NLUPortal, util, ImageEditDialogueSystem


app = Flask(__name__, template_folder='../app/template',
            static_folder='../app/static')


################
#    System    #
################
DialogueSystem = None

# global session
# session = util.SessionPortal(config["session"])


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/step", methods=["POST"])
def step():
    # Load sesssion
    session_id = int(request.form.get("session_id", 0))  # default to 0
    print("session_id", session_id)
    # dialogue = session.retrieve(session_id)
    # system.state.from_json(dialogue["system_state"])
    # photoshop.from_json(dialogue["photoshop_state"])

    # Continue doing what's supposed to be done
    user_utt = request.form.get('user_utterance', '')
    print("usr:", user_utt)

    sys_utt = DialogueSystem.step(user_utt)
    print("sys:", sys_utt)

    img = DialogueSystem.photoshop.get_image()
    b64_img_str = util.img_to_b64(img)

    # Record to session
    # turn_info = {"user": user_utt,
    #             "system": sys_utt, "turn": system.turn_id}
    # state_json = system.state.to_json()
    # ps_json = photoshop.to_json()
    # session.add_turn(session_id, state_json, ps_json, turn_info)

    # Create return_object
    obj = {}
    obj['system_utterance'] = sys_utt
    obj['b64_img_str'] = b64_img_str
    return jsonify(obj)


def serve(args):
    global DialogueSystem
    DialogueSystem = ImageEditDialogueSystem(args.config)
    DialogueSystem.reset()

    image_path = "example/COCO_train2014_000000229598.jpg"
    DialogueSystem.open(image_path)

    app.run(host='0.0.0.0', port=2000, debug=args.debug)


def terminal(args):

    session_id = int(time.time())

    DialogueSystem = ImageEditDialogueSystem(args.config)
    DialogueSystem.reset()

    # Open an image
    image_path = "example/COCO_train2014_000000229598.jpg"
    DialogueSystem.open(image_path)
    # result, msg = photoshop.control("open", {'image_path': image_path})

    # assert result
    # turn_info = {"turn": 0, "agenda_idx": 10}
    # session.add_turn(session_id, system.state.to_json(), photoshop.to_json(), turn_info)

    while True:
        # Load from session
        # logger.info("Loading from session {}".format(session_id))
        # dialogue = session.retrieve(session_id)
        # system.state.from_json(dialogue["system_state"])
        # photoshop.from_json(dialogue["photoshop_state"])

        user_utt = input("User: ")

        sys_utt = DialogueSystem.step(user_utt)

        print("System", sys_utt)

        # Save to session
        # logger.info("Saving session")
        # turn_info = {"user": user_utt, "system": sys_utt, "turn": system.turn_id}
        # state_json = system.state.to_json()
        # ps_json = photoshop.to_json()
        # session.add_turn(session_id, state_json, ps_json, turn_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default="terminal")
    parser.add_argument('-c', '--config', type=str, default="./config/deploy/rule.json",
                        help="Path to deployment config")
    parser.add_argument('-p', '--port', type=int,
                        default=2000, help="server port")
    parser.add_argument('-d', '--debug', action="store_true",
                        help="Flask server debug mode")
    args = parser.parse_args()

    print("mode:", args.mode)
    # Load dialogue system
    # load_dialoguesystem(args)

    mode = args.mode
    if mode == "terminal":
        terminal(args)
    elif mode == "serve":
        serve(args)
