import argparse
import json
import logging
import os
import random
import time
import uuid

from flask import Flask, abort, jsonify, redirect, render_template, request

from cie import ImageEditRealUserInterface, SystemAct, VisionEnginePortal, util

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
Config = None
SessionManager = None
VisionEngine = None
ImageDir = None


def generate_session_id():
    session_id = str(uuid.UUID(bytes=os.urandom(16)))
    return session_id


def get_image_path(image_id):
    image_name = image_id + ".jpg"
    image_path = os.path.join(ImageDir, image_name)
    return image_path


def get_random_image_id():
    image_names = os.listdir(ImageDir)
    image_name = random.choice(image_names)
    image_id = image_name.rstrip(".jpg")
    return image_id


@app.route("/")
def index():
    # Gets image_id
    image_id = request.args.get('image_id', get_random_image_id())
    image_path = get_image_path(image_id)
    if not os.path.exists(image_path):
        abort(404)

    # Generates session id
    session_id = generate_session_id()

    # Creates session
    print("[index] session", session_id)
    SessionManager.create_session(session_id)

    # Create System
    DialogueSystem = ImageEditRealUserInterface(Config)
    DialogueSystem.reset()

    # Open image
    DialogueSystem.open(image_path)
    SessionManager.add_image_id(session_id, image_id)

    # Create initial state
    system_state = DialogueSystem.to_json()
    SessionManager.add_turn(session_id, system_state)

    return render_template('index.html', session_id=session_id)


@app.route("/init", methods=["POST"])
def init():
    session_id = request.form.get('session_id', "-1")
    print("[init] session", session_id)

    # Create System
    DialogueSystem = ImageEditRealUserInterface(Config)
    DialogueSystem.reset()

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
    session_id = request.form.get("session_id", "-1")  # default to 0
    print("[step] session_id", session_id)

    # Create System
    DialogueSystem = ImageEditRealUserInterface(Config)
    DialogueSystem.reset()

    # Retreive last state
    last_system_state = SessionManager.retrieve(session_id)
    DialogueSystem.from_json(last_system_state)

    user_utterance = request.form.get('user_utterance', '')
    user_act = {'user_utterance': user_utterance}
    print("User:", user_utterance)

    prev_identified_object = DialogueSystem.manager.state.get_slot(
        'object').get_max_value()

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

    identified_object = DialogueSystem.manager.state.get_slot(
        'object').get_max_value()

    # Create return_object
    obj = {}
    obj['system_utterance'] = sys_utt
    obj['b64_img_str'] = b64_img_str
    obj['system_act'] = sys_act['system_acts'][0]
    obj['object'] = identified_object or prev_identified_object
    return jsonify(obj)


@app.route("/survey", methods=["POST"])
def survey():
    session_id = request.form.get("session_id", "-1")
    survey = json.loads(request.form.get("survey", {}))

    print('[survey] session_id', session_id)
    print("survey", survey)

    SessionManager.add_survey(session_id, survey)

    obj = {}
    return jsonify(obj)


def serve(args):
    """
    Load real global arguments
    """
    global Config
    Config = args.config

    global SessionManager
    SessionManager = util.PickleManager(args.session_dir)

    global VisionEngine
    config_json = util.load_from_json(args.config)
    VisionEngine = VisionEnginePortal(config_json['agents']['visionengine'])

    global ImageDir
    ImageDir = config_json['image_dir']

    app.run(host='0.0.0.0', port=2000, debug=args.debug)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./config/deploy/rule.json",
                        help="Path to deployment config")
    parser.add_argument('-s', '--session-dir', type=str, default="./pickled",
                        help="Path to session pickle directory")
    parser.add_argument('-p', '--port', type=int,
                        default=2000, help="server port")
    parser.add_argument('-d', '--debug', action="store_true",
                        help="Flask server debug mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    serve(args)
