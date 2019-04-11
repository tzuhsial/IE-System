import json
import logging
import os
import pprint
import random
import sys
import time
from configparser import ConfigParser
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, root_dir)

pp = pprint.PrettyPrinter(indent=2)

import numpy as np
from flask import Flask, jsonify, request, render_template


from cie import TrackerPortal, SystemPortal, PhotoshopPortal
from cie import util


# Log outputs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def serve(argv):
    config_file = argv[2]
    config = util.load_from_json(config_file)
    test_agendas = util.load_from_pickle(config["agendas"]["test"])

    # Load Session Manager
    session = util.SessionPortal(config["session"])

    # Load agents here
    agents_config = config["agents"]
    tracker = TrackerPortal(agents_config["tracker"])
    system = SystemPortal(agents_config["system"])
    photoshop = PhotoshopPortal(agents_config["photoshop"])

    tracker.reset()
    system.reset()
    photoshop.reset()

    # For RL Policies
    policy_config = config["agents"]["system"]["policy"]
    if policy_config["name"] == "DQNPolicy":
        assert policy_config["load"] is not None
        system.policy.update_epsilon(test=True)

    app = Flask(
        __name__, template_folder='../app/template', static_folder='../app/static')

    @app.route("/")
    def index():
        return render_template('index.html')

    @app.route("/sample", methods=["POST"])
    def sample():
        """
        Sample a goal and an image
        """
        # Load sesssion
        session_id = int(request.form.get("session_id", 0))  # default to 0
        goal_idx = int(request.form.get("goal_idx", -1))
        print("session_id", session_id)
        dialogue = session.retrieve(session_id)

        # Action
        tracker.reset()
        system.reset()
        photoshop.reset()

        if goal_idx >= 0:
            idx, agenda = goal_idx, test_agendas[goal_idx]
        else:
            idx, agenda = random.choice(list(enumerate(test_agendas)))

        open_goal = agenda[0]
        open_slots = open_goal['slots']

        image_path_slot = util.find_slot_with_key("image_path", open_slots)
        image_path = image_path_slot['value']
        image_name = os.path.basename(image_path)
        image_dir = "./sampled_100/image"
        image_path = os.path.join(image_dir, image_name)
        print('image_path:', image_path)
        result, msg = photoshop.control("open", {'image_path': image_path})
        assert result

        img = photoshop.get_image()
        if img is None:
            loaded_b64_img_str = ""
        else:
            loaded_b64_img_str = util.img_to_b64(img)

        photoshop_act = photoshop.act()
        # Intent adjust
        system.observe(photoshop_act)
        system_act = system.act()

        # Save to session
        turn_info = {"agenda_id": idx, "turn": 0}
        state_json = system.state.to_json()
        ps_json = photoshop.to_json()
        session.add_turn(session_id, state_json, ps_json, turn_info)
        session.add_policy(session_id, system.policy.__class__.__name__)

        # Build return object
        goal = {}
        for slot in agenda[1]["slots"]:
            goal[slot["slot"]] = slot["value"]

        goal["intent"] = agenda[1]["intent"]["value"]
        object_mask = util.b64_to_img(goal["object_mask_str"])
        object_mask_img = object_mask

        object_mask_img_str = util.img_to_b64(object_mask_img)
        goal["object_mask_img_str"] = object_mask_img_str

        obj = {}
        obj["b64_img_str"] = loaded_b64_img_str
        obj["goal"] = goal
        return jsonify(obj)

    @app.route("/step", methods=["POST"])
    def step():
        # Load sesssion
        session_id = int(request.form.get("session_id", 0))  # default to 0
        print("session_id", session_id, "step")
        dialogue = session.retrieve(session_id)
        system.state.from_json(dialogue["system_state"])
        photoshop.from_json(dialogue["photoshop_state"])

        # Continue doing what's supposed to be done
        user_utt = request.form.get('user_utterance', '')
        print("user_utterance:", user_utt)

        click_coordinates = request.form.get("gesture_click", None)
        box_coordinates = request.form.get("object_mask_str", None)

        user_act = {
            "user_utterance": user_utt,
        }

        tracker.observe(user_act)
        tracker_act = tracker.act()

        # gesture_click
        if click_coordinates is not None:
            click_coordinates = json.loads(click_coordinates)
            x = click_coordinates['x']
            y = click_coordinates['y']

            # Create gesture_click_str
            img = photoshop.get_image()
            gesture_click = np.zeros_like(img)
            gesture_click[x][y][:] = 255
            # Perhaps we can expand a little bit
            gesture_click_str = util.img_to_b64(gesture_click)

            gesture_click_slot = {'slot': 'gesture_click',
                                  'value': gesture_click_str, 'conf': 1.0}

            tracker_act["user_acts"][0]['slots'].append(gesture_click_slot)

        # object_mask_str
        if box_coordinates is not None:
            box_coordinates = json.loads(box_coordinates)
            y = box_coordinates["left"]
            x = box_coordinates["top"]
            width = box_coordinates["width"]
            height = box_coordinates["height"]

            object_mask = np.zeros_like(photoshop.get_image())
            object_mask[x:x+height, y:y+width] = 255

            object_mask_str = util.img_to_b64(object_mask)
            object_mask_str_slot = {'slot': "object_mask_str",
                                    'value': object_mask_str, 'conf': 1.0}

            tracker_act["user_acts"][0]['slots'].append(object_mask_str_slot)

        pp.pprint(tracker_act["user_acts"][0])

        system.observe(tracker_act)
        system_act = system.act()

        # Check for Query Option
        sys_da = system_act['system_acts'][0]['dialogue_act']['value']
        sys_utt = system_act["system_utterance"]
        if sys_da == "query":

            mask_str_node = system.state.get_slot("object_mask_str")
            nresult = len(mask_str_node.value_conf_map)
            sys_utt += " Found {} results. ".format(nresult)

            # We don't need to wait for another user utterance
            # Load the query results to photoshop
            mask_strs = []
            for mask_idx, mask_str in enumerate(mask_str_node.value_conf_map.keys()):
                mask_strs.append((mask_idx, mask_str))
            result, msg = photoshop.control(
                'load_mask_strs', {"mask_strs": mask_strs})

            # Perform a 2nd action
            system.observe({})
            system_act = system.act()

            sys_utt += system_act["system_utterance"]

        else:
            # Always load mask_strs
            mask_str_node = system.state.get_slot("object_mask_str")
            mask_strs = []
            for mask_idx, mask_str in enumerate(mask_str_node.value_conf_map.keys()):
                mask_strs.append((mask_idx, mask_str))

            result, msg = photoshop.control(
                'load_mask_strs', {"mask_strs": mask_strs})

        print("System:", sys_utt)
        photoshop.observe(system_act)
        photoshop_act = photoshop.act()

        img = photoshop.get_image()
        if img is None:
            loaded_b64_img_str = ""
        else:
            loaded_b64_img_str = util.img_to_b64(img)

        system.observe(photoshop_act)

        # Record to session
        turn_info = {"user": user_utt,
                     "system": sys_utt, "turn": system.turn_id}
        state_json = system.state.to_json()
        ps_json = photoshop.to_json()
        session.add_turn(session_id, state_json, ps_json, turn_info)

        # Create return_object
        obj = {}
        obj['system_utterance'] = sys_utt
        obj['b64_img_str'] = loaded_b64_img_str  # We need to return the image
        obj['last_execute_result'] = photoshop.last_execute_result
        return jsonify(obj)

    @app.route("/result", methods=["POST"])
    def results():
        # Load sesssion
        print("result")
        session_id = int(request.form.get("session_id", 0))  # default to 0
        result = request.form.get("result", None)

        session.add_result(session_id, result)

        dialogue = session.retrieve(session_id)
        print('turns', dialogue['turns'])

        # Create return_object
        obj = {}
        obj["system_utterance"] = "Result recorded. Thank you for participating!"
        return jsonify(obj)

    @app.route("/reset", methods=["POST"])
    def reset():
        # Load sesssion
        print("reset")
        session_id = int(request.form.get("session_id", 0))  # default to 0
        print("session_id", session_id, "reset")
        dialogue = session.retrieve(session_id)
        print('dialogue turns', dialogue['turns'])
        system.state.from_json(dialogue["system_state"])
        photoshop.from_json(dialogue["photoshop_state"])

        tracker.reset()
        system.reset()
        photoshop.reset()

        img = photoshop.get_image()
        if img is None:
            loaded_b64_img_str = ""
        else:
            loaded_b64_img_str = util.img_to_b64(img)

        photoshop_act = photoshop.act()
        system.observe(photoshop_act)
        system_act = system.act()

        # Save to session
        turn_info = {"reset": True}
        state_json = system.state.to_json()
        ps_json = photoshop.to_json()
        session.add_turn(session_id, state_json, ps_json, turn_info)

        # Create return_object
        obj = {}
        obj['system_utterance'] = "Welcome to Wonderland!"
        obj['b64_img_str'] = loaded_b64_img_str
        return jsonify(obj)

    app.run(host='0.0.0.0', port=2000, debug=True)


def terminal(argv):
    config_file = argv[2]
    config = util.load_from_json(config_file)

    session = util.SessionPortal(config["session"])
    session_id = int(time.time())

    #
    agendas = util.load_from_pickle(config["agendas"]["test"])

    # Load agents here
    agents_config = config["agents"]
    tracker = TrackerPortal(agents_config["tracker"])
    system = SystemPortal(agents_config["system"])
    photoshop = PhotoshopPortal(agents_config["photoshop"])

    # Initialize session
    tracker.reset()
    system.reset()
    photoshop.reset()

    agenda = agendas[10]
    open_goal = agenda[0]

    open_goal = agenda[0]
    open_slots = open_goal['slots']

    image_path_slot = util.find_slot_with_key("image_path", open_slots)
    image_path = image_path_slot['value']
    image_dir = "./sampled_100/image"
    image_name = os.path.basename(image_path)
    image_path = os.path.join(image_dir, image_name)

    result, msg = photoshop.control("open", {'image_path': image_path})
    assert result
    turn_info = {"turn": 0, "agenda_idx": 10}
    session.add_turn(session_id, system.state.to_json(),
                     photoshop.to_json(), turn_info)
    photoshop_act = {}
    while True:
        # Load from session
        logger.info("Loading from session {}".format(session_id))
        dialogue = session.retrieve(session_id)
        system.state.from_json(dialogue["system_state"])
        photoshop.from_json(dialogue["photoshop_state"])

        user_utt = input("User: ")

        user_act = {"user_utterance": user_utt}

        tracker.observe(user_act)
        tracker_act = tracker.act()

        pp.pprint(tracker_act["user_acts"][0])

        system.observe(photoshop_act)
        system.observe(tracker_act)
        system_act = system.act()
        sys_utt = system_act["system_utterance"]
        print("System:", sys_utt)
        photoshop.observe(system_act)
        photoshop_act = photoshop.act()

        # Save to session
        logger.info("Saving session")
        turn_info = {"user": user_utt,
                     "system": sys_utt, "turn": system.turn_id}
        state_json = system.state.to_json()
        ps_json = photoshop.to_json()
        session.add_turn(session_id, state_json, ps_json, turn_info)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "terminal":
        terminal(sys.argv)
    elif mode == "serve":
        serve(sys.argv)
