import argparse
import json
import logging
import os
import pprint
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

from cie import ImageEditRealUserInterface, SystemAct, VisionEnginePortal, util

pp = pprint.PrettyPrinter(indent=4)

app = Flask(__name__, template_folder='./app/template',
            static_folder='./app/static')

# Logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logFormatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


# Global Variables
DialogueSystem = None
SessionManager = None
VisionEngine = None


@app.route("/")
def index():
    # Assign session_id and image_path
    # Default values for debugging purposes

    session_id = request.form.get('session_id', None)
    if session_id is None:
        session_id = int(time.time()) # Randomly generate using timestamp
    else:
        session_id = int(session_id)
        
    image_path = request.form.get(
        'image_path', "example/COCO_train2014_000000229598.jpg")
    
    # Creates session
    SessionManager.create_session(session_id)

    # Adds to
    DialogueSystem.open(image_path)
    system_state = DialogueSystem.to_json()
    SessionManager.add_turn(session_id, system_state)

    return render_template('index.html', session_id=session_id)

@app.route("/init", methods=["POST"])
def init():
    session_id = int(request.form.get('session_id', 0))
    print("session", session_id)

    last_system_state = SessionManager.retrieve(session_id)
    DialogueSystem.from_json(last_system_state)

    img = DialogueSystem.get_image()
    b64_img_str = util.img_to_b64(img)

    # Create return_object
    obj = {}
    obj['system_utterance'] = "Hi! This is an image editing chatbot.  How may I help you?"
    obj['b64_img_str'] = b64_img_str
    return jsonify(obj)


@app.route("/step", methods=["POST"])
def step():
    # Load sesssion
    session_id = int(request.form.get("session_id", 0))  # default to 0
    print("session_id", session_id)

    last_system_state = SessionManager.retrieve(session_id)
    DialogueSystem.from_json(last_system_state)

    user_utterance = request.form.get('user_utterance', '')
    user_act = {'user_utterance': user_utterance}
    print("User:", user_utterance)

    DialogueSystem.observe(user_act)
    sys_act = DialogueSystem.act()

    sys_utt = sys_act['system_utterance']
    print("System:", sys_utt)

    # Store system state
    SessionManager.add_turn(session_id, DialogueSystem.to_json())

    sys_dialogue_act = sys_act['system_acts'][0]['dialogue_act']['value']
    if sys_dialogue_act == SystemAct.QUERY:
        # Query Vision Engine
        VisionEngine.observe(sys_act)
        vis_act = VisionEngine.act()
        print("Vision:", vis_act['visionengine_utterance'])

        DialogueSystem.observe(vis_act)
        sys_act = DialogueSystem.act()
        sys_utt = sys_act['system_utterance']
        print("System:", sys_utt)

        # Store system state
        SessionManager.add_turn(session_id, DialogueSystem.to_json())

    img = DialogueSystem.get_image()
    b64_img_str = util.img_to_b64(img)

    # Create return_object
    obj = {}
    obj['system_utterance'] = sys_utt
    obj['b64_img_str'] = b64_img_str
    return jsonify(obj)


def serve(args):
    global DialogueSystem
    DialogueSystem = ImageEditRealUserInterface(args.config)
    DialogueSystem.reset()

    global SessionManager
    SessionManager = util.PickleManager('pickled')

    global VisionEngine
    config_json = util.load_from_json(args.config)
    VisionEngine = VisionEnginePortal(config_json['agents']['visionengine'])

    app.run(host='0.0.0.0', port=2000, debug=args.debug)


def terminal(args):
    """
    Terminal mode: used for debugging purposes
    """
    # Initialize session
    SessionManager = util.PickleManager('pickled')

    session_id = int(time.time())
    SessionManager.create_session(session_id)

    # Create system
    DialogueSystem = ImageEditRealUserInterface(args.config)
    DialogueSystem.reset()

    # Open an image
    image_path = "example/COCO_train2014_000000229598.jpg"
    DialogueSystem.open(image_path)
    # result, msg = photoshop.control("open", {'image_path': image_path})

    # assert result
    system_state = DialogueSystem.to_json()
    SessionManager.add_turn(session_id, system_state)

    # Sanity Check
    DialogueSystem.from_json(system_state)
    system_state2 = DialogueSystem.to_json()
    assert system_state == system_state2

    # Load VisionEngine
    config_json = util.load_from_json(args.config)
    VisionEngine = VisionEnginePortal(config_json['agents']['visionengine'])

    turn = 1
    for usr_utt in ["adjust brightness of man on left by 50", "adjust contrast of cake in front by -50"]:

        print('goal', usr_utt)
        sys_dialogue_act = SystemAct.GREETING
        no_and_yes = ["no.", "yes."]

        while True:
            print('turn', turn)
            # Load from session
            logger.info("Loading from session {}".format(session_id))
            last_system_state = SessionManager.retrieve(session_id)
            DialogueSystem.from_json(last_system_state)

            if sys_dialogue_act == SystemAct.EXECUTE:
                break
            elif sys_dialogue_act == SystemAct.CONFIRM:
                usr_or_vis_act = {"user_utterance": no_and_yes[0]}
                no_and_yes.pop(0)
                print('User:', usr_or_vis_act['user_utterance'])
            elif sys_dialogue_act == SystemAct.QUERY:
                # Send to Vision Engine
                VisionEngine.observe(sys_act)
                usr_or_vis_act = VisionEngine.act()
                print("Visionengine:",
                      usr_or_vis_act['visionengine_utterance'])
            else:
                # usr_utt = input("User: ")
                usr_or_vis_act = {"user_utterance": usr_utt}
                print('User:', usr_or_vis_act['user_utterance'])

            DialogueSystem.observe(usr_or_vis_act)
            sys_act = DialogueSystem.act()
            sys_utt = sys_act['system_utterance']

            print("System:", sys_utt)

            sys_dialogue_act = sys_act['system_acts'][0]['dialogue_act']['value']

            image = DialogueSystem.get_image()

            util.imwrite(image, "example/terminal2/{}.png".format(turn))
            turn += 1

            # Save to session
            current_session_state = DialogueSystem.to_json()
            SessionManager.add_turn(session_id, current_session_state)


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

    mode = args.mode
    if mode == "terminal":
        terminal(args)
    elif mode == "serve":
        serve(args)
