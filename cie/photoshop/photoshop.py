import json
import logging
import os
import requests
from urllib.parse import urljoin

from .sps import SimplePhotoshop

from ..core import SystemAct, PhotoshopAct
from ..util import find_slot_with_key, img_to_b64, build_slot_dict, slots_to_args, imread

logger = logging.getLogger(__name__)


def PhotoshopPortal(photoshop_config):

    photoshop_type = photoshop_config["photoshop"]
    use_client = photoshop_config['client']
    uri = photoshop_config['uri']

    if photoshop_type == "SuperficialPhotoshop":
        return SuperficialPhotoshopAgent()
    elif photoshop_type == "SimplePhotoshop":
        if not use_client:
            return SimplePhotoshopAgent()
        else:
            return SimplePhotoshopClient(uri)
    else:
        raise NotImplementedError


class SuperficialPhotoshopAgent(object):
    """
    Doesn't perform any real image edits
    """

    def __init__(self):
        # Define here for now
        self.action_slot_dict = {
            "open": ["image_path"],
            "adjust": ["object_mask_str", "attribute", "adjust_value"],
            "redo": [],
            "close": [],
            "undo": []
        }

    def reset(self):
        self.observation = {}
        self.last_execute_result = False
        self.original_b64_img_str = ""
        # Simulate the history
        self.num_edits = -1
        self.ptr = -1

    def observe(self, observation):
        """
        Observes system act
        """
        self.observation = observation

    def act(self):
        system_acts = self.observation.get('system_acts', list())
        sys_act = system_acts[0]
        sys_dialogue_act = sys_act['dialogue_act']['value']

        if sys_dialogue_act == SystemAct.EXECUTE:
            intent = sys_act['intent']['value']
            slots = sys_act['slots']
            self.act_execute(intent, slots)

        ps_act = self.act_inform()

        photoshop_act = {}
        photoshop_act['photoshop_acts'] = [ps_act]

        self.observation = {}
        return photoshop_act

    def act_inform(self):
        """
        Creates observation for user & system,
        User:
            execute_result
        System:
            original_b64_img_str
            has_previous_history
            has_next_history
        """
        slots = []
        exec_result_slot = build_slot_dict('execute_result',
                                           self.last_execute_result, 1.0)
        original_b64_img_str = build_slot_dict('original_b64_img_str',
                                               self.original_b64_img_str, 1.0)
        has_previous_history = build_slot_dict('has_previous_history',
                                               self.ptr > 0, 1.0)
        has_next_history = build_slot_dict('has_next_history',
                                           self.ptr < self.num_edits, 1.0)

        slots.append(exec_result_slot)
        slots.append(original_b64_img_str)
        slots.append(has_previous_history)
        slots.append(has_next_history)
        ps_act = {}
        ps_act['dialogue_act'] = build_slot_dict('dialogue_act', "inform", 1.0)
        ps_act['slots'] = slots
        return ps_act

    def act_execute(self, intent, slots):

        execute_result = True
        if intent in ["open", "adjust"]:
            required_slot_names = self.action_slot_dict[intent]
            for slot_name in required_slot_names:
                s = find_slot_with_key(slot_name, slots)
                if not s or not s.get('value'):
                    execute_result = False
                    break

            # Needs to open an image before adjusting
            if intent == "adjust" and self.original_b64_img_str is None:
                execute_result = False

            if intent == "open" and execute_result:
                assert slots[0]["slot"] == "image_path"
                image_path = slots[0]['value']
                if not os.path.exists(image_path):
                    raise ValueError
                image = imread(image_path)
                b64_img_str = img_to_b64(image)
                self.original_b64_img_str = b64_img_str

            if execute_result:
                self.ptr += 1
                self.num_edits += 1

        elif intent == "undo":
            if self.ptr <= 0:
                execute_result = False
            else:
                self.ptr -= 1

        elif intent == "redo":
            if self.ptr >= self.num_edits:
                execute_result = False
            else:
                self.ptr += 1

        elif intent == "close":
            execute_result = True

        self.last_execute_result = execute_result


class SimplePhotoshopAgent(SimplePhotoshop):
    """
    An agent that loads SimplePhotoshop directly
    """

    def __init__(self):
        """
        Calls parent class for initialization
        """
        super(SimplePhotoshopAgent, self).__init__()

    def reset(self):
        """
        Reset
        """
        self.observation = {}
        self.masks.clear()
        self.last_execute_result = False
        self.last_execute_message = ""

    def observe(self, observation):
        """
        Observes system act
        """
        self.observation = observation

    def act(self):
        """
        Photoshop actions
        """
        # Get all system actions
        system_acts = self.observation.get('system_acts', list())
        sys_act = system_acts[0]
        sys_dialogue_act = sys_act['dialogue_act']['value']

        if sys_dialogue_act == SystemAct.EXECUTE:
            intent = sys_act['intent']['value']
            slots = sys_act['slots']
            self.act_execute(intent, slots)

        ps_act = self.act_inform()

        photoshop_act = {}
        photoshop_act['photoshop_acts'] = [ps_act]

        self.observation = {}
        return photoshop_act

    def act_execute(self, intent, slots):

        if intent in PhotoshopAct.control_acts():
            action = "control"
        else:
            action = "edit"

        # Build data
        args = slots_to_args(slots)

        if action == "control":
            execute_result, msg = self.control(intent, args)
        else:
            execute_result, msg = self.execute(intent, args)

        self.last_execute_result = execute_result
        self.last_execute_message = msg

        return execute_result, msg

    def act_inform(self):
        """
        Informs the user, basically what the user sees.
        b64_img_str
        masked_b64_img_str
        mask_strs 
        """
        # Get slots
        ps_act = {}
        slots = []
        if self.get_image(False) is None:
            original_b64_img_str = ""
            b64_img_str = ""
            masked_b64_img_str = ""

        else:
            original_image = self.background
            original_b64_img_str = img_to_b64(original_image)

            image = self.get_image(False)
            b64_img_str = img_to_b64(image)

            masked_image = self.get_image(True)
            masked_b64_img_str = img_to_b64(masked_image)

        original_b64_img_str_slot = build_slot_dict('original_b64_img_str',
                                                    original_b64_img_str, 1.0)
        b64_img_str_slot = build_slot_dict('b64_img_str', b64_img_str, 1.0)
        masked_b64_img_str_slot = build_slot_dict('masked_b64_img_str',
                                                  masked_b64_img_str, 1.0)

        mask_strs = []
        for mask_idx, mask in self.get_masks():
            mask_str = img_to_b64(mask)
            mask_strs.append((mask_idx, mask_str))

        mask_strs_slot = build_slot_dict('mask_strs', mask_strs, 1.0)

        exec_result_slot = build_slot_dict("execute_result",
                                           self.last_execute_result, 1.0)

        slots.append(original_b64_img_str_slot)
        slots.append(b64_img_str_slot)
        slots.append(masked_b64_img_str_slot)
        slots.append(mask_strs_slot)
        slots.append(exec_result_slot)

        ps_act = {
            'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
            'slots': slots
        }

        return ps_act


class SimplePhotoshopClient(object):
    """
    Client interface to SimplePhotoshop
    """

    def __init__(self, photoshop_uri):
        """
        """
        # Configuration
        self.photoshop_uri = photoshop_uri
        self.action_uri = urljoin(photoshop_uri, 'action')

        self.observation = {}

    @property
    def name(self):
        return self.__class__.__name__

    def reset(self):
        self.observation = {}

    def observe(self, observation):
        self.observation = observation

    def act(self):
        """
        POST load_mask_strs if any of photoshop actions belong to load_mask_strs
        POST action if EXECUTE is observed
        """
        # Get all system actions
        system_acts = self.observation.get('system_acts', list())

        for sys_act in system_acts:
            sys_dialogue_act = sys_act['dialogue_act']['value']

            # Load mask_strs action
            if sys_dialogue_act in SystemAct.load_mask_strs_acts():

                # Get all mask_str slots from mask, or could be empty
                mask_str_slots = []
                for slot in sys_act['slots']:
                    # is not a request
                    if slot['slot'] == "object_mask_str" and slot.get('value'):
                        mask_str_slots.append(slot)

                # Process mask_str_slot to SimplePhotoshop arg format
                mask_strs = []
                for idx, mask_slot in enumerate(mask_str_slots):
                    tup = (str(idx), mask_slot['value'])
                    mask_strs.append(tup)

                args = {}
                args['mask_strs'] = mask_strs

                data = {}
                data['action'] = 'control'
                data['intent'] = PhotoshopAct.LOAD_MASK_STRS
                data['args'] = json.dumps(args)

                logger.info("POST {} Number of masks {}".format(
                    PhotoshopAct.LOAD_MASK_STRS, len(mask_strs)))
                res = requests.post(self.action_uri, data=data)
                res.raise_for_status()

            # Execute actions
            if sys_dialogue_act == SystemAct.EXECUTE:

                # control_type & edit_type
                intent = sys_act['intent']['value']

                if intent in PhotoshopAct.control_acts():
                    action = "control"
                else:
                    action = "edit"

                # Build data
                args = {}
                for slot in sys_act['slots']:
                    # Exclude mask_strs
                    if slot['slot'] != 'mask_str':
                        slot_name = slot.get('slot')
                        slot_value = slot.get('value')
                        args[slot_name] = slot_value

                data = {}
                data['action'] = action
                data['intent'] = intent
                data['args'] = json.dumps(args)

                # Execute action
                logger.info("POST {}".format(intent))
                res = requests.post(self.action_uri, data=data)
                res.raise_for_status()

        # Retrieve Image
        data = {'action': 'control', 'intent': 'check', 'args': json.dumps({})}
        res = requests.post(self.action_uri, data=data)
        res.raise_for_status()
        check_obj = res.json()

        b64_img_str = check_obj.get("b64_img_str", "")
        masked_b64_img_str = check_obj.get("masked_b64_img_str", "")

        # Build return object
        photoshop_act = {}
        ps_act = {
            'dialogue_act':
            build_slot_dict('dialogue_act', 'inform', 1.0),
            'slots': [
                build_slot_dict('b64_img_str', b64_img_str, 1.0),
                build_slot_dict('masked_b64_img_str', masked_b64_img_str, 1.0)
            ]
        }

        photoshop_act = {}
        photoshop_act['photoshop_acts'] = [ps_act]
        self.observation = {}
        return photoshop_act
