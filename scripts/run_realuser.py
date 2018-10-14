import json
import logging
import os
import pprint
import random
import sys
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
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def serve(argv):
    config_file = argv[2]
    config = util.load_from_json(config_file)
    test_agendas = util.load_from_pickle(config["agendas"]["test"])

    # Load agents here
    agents_config = config["agents"]
    tracker = TrackerPortal(agents_config["tracker"])
    system = SystemPortal(agents_config["system"])
    photoshop = PhotoshopPortal(agents_config["photoshop"])

    tracker.reset()
    system.reset()
    photoshop.reset()

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
        tracker.reset()
        system.reset()
        photoshop.reset()

        idx, agenda = random.choice(list(enumerate(test_agendas)))
        #agenda = test_agendas[10]

        open_goal = agenda[0]
        open_slots = open_goal['slots']

        image_path_slot = util.find_slot_with_key("image_path", open_slots)

        image_path = image_path_slot['value']
        result, msg = photoshop.control("open", {'image_path': image_path})
        assert result

        img = photoshop.get_image()
        if img is None:
            loaded_b64_img_str = ""
        else:
            loaded_b64_img_str = util.img_to_b64(img)

        # Intent adjust

        system.observe({})
        system_act = system.act()

        # Build return object

        goal = {}
        for slot in agenda[1]["slots"]:
            goal[slot["slot"]] = slot["value"]

        goal["intent"] = agenda[1]["intent"]["value"]

        # object_mask_str
        object_mask = util.b64_to_img(goal["object_mask_str"])
        object_mask_img = object_mask

        object_mask_img_str = util.img_to_b64(object_mask_img)
        goal["object_mask_img_str"] = object_mask_img_str

        obj = {}
        obj["b64_img_str"] = loaded_b64_img_str
        obj["goal"] = goal

        return jsonify(obj)

    @app.route("/open", methods=["POST"])
    def open():
        """ Opens an image
        """
        b64_img_str = request.form.get("b64_img_str", "")
        args = {"b64_img_str":  b64_img_str}
        result, msg = photoshop.control("load", args)
        print("result", result)

        img = photoshop.get_image()
        if img is None:
            loaded_b64_img_str = ""
        else:
            loaded_b64_img_str = util.img_to_b64(img)

        # Build return object
        obj = {}
        obj["b64_img_str"] = loaded_b64_img_str
        return jsonify(obj)

    @app.route("/step", methods=["POST"])
    def step():

        user_utterance = request.form.get('user_utterance', '')
        print("user_utterance:", user_utterance)

        click_coordinates = request.form.get("gesture_click", None)
        box_coordinates = request.form.get("object_mask_str", None)

        user_act = {
            "user_utterance": user_utterance,
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
            assert result, "load_mask_strs failure in SimplePhotoshop"

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

        # Create return_object
        obj = {}
        obj['system_utterance'] = sys_utt
        obj['b64_img_str'] = loaded_b64_img_str  # We need to return the image
        return jsonify(obj)

    @app.route("/reset", methods=["POST"])
    def reset():
        tracker.reset()
        system.reset()
        photoshop.reset()

        img = photoshop.get_image()
        if img is None:
            loaded_b64_img_str = ""
        else:
            loaded_b64_img_str = util.img_to_b64(img)

        # Create return_object
        obj = {}
        obj['system_utterance'] = "Welcome to Wonderland!"
        obj['b64_img_str'] = loaded_b64_img_str
        return jsonify(obj)

    app.run(host='0.0.0.0', port=2000, debug=True)


def terminal(argv):
    config_file = argv[2]
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
                    # "episode_done": False,
                    # "reward": 0
                    }

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
    mode = sys.argv[1]
    if mode == "terminal":
        terminal(sys.argv)
    elif mode == "serve":
        serve(sys.argv)
