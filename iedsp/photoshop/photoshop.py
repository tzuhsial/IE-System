import json
import requests

from ..ontology import Ontology
from ..util import find_slot_with_key


class SimplePhotoshopAPI(object):
    """
        Simple Photoshop Backend Agent
    """

    def __init__(self, api_url="http://localhost:2005"):
        """Stores the api for connection
        """
        self.edit_url = api_url + "/edit"
        self.control_url = api_url + "/control"
        self.check_url = api_url + "/check"

        self.observation = {}

    @property
    def name(self):
        return self.__class__.__name__

    def reset(self):
        self.observation = {}

    def observe(self, observation):
        self.observation = observation

    def act(self):
        """ Executes action when system dialogue act "execute" is observed
        """
        # Get all system actions
        backend_act = {}
        system_acts = self.observation.get('system_acts', list())
        for sys_act in system_acts:
            sys_dialogue_act = sys_act['dialogue_act']

            if sys_dialogue_act == "execute":

                slots = sys_act['slots']

                # Edit or Control
                intent_idx, intent_slot = find_slot_with_key(
                    'intent', slots)
                intent_type = intent_slot['value']
                if intent_type in ["open", "load", "select_object", "select_object_mask_id", "redo", "undo"]:
                    intent_type_key = 'control_type'
                    url = self.control_url
                elif intent_type in ["adjust"]:
                    intent_type_key = 'edit_type'
                    url = self.edit_url
                else:
                    raise ValueError(
                        "Unknown intent_type: {}".format(intent_type))

                slot_names = Ontology.get(intent_type)

                # Build post data to backend
                data = {}
                data[intent_type_key] = intent_type
                args = {}
                for slot_name in slot_names:
                    _, slot = find_slot_with_key(slot_name, slots)
                    args[slot_name] = slot['value']

                # Find object_mask_id if any
                mask_id_idx, mask_id_slot = find_slot_with_key(
                    'object_mask_id', slots)
                if mask_id_idx >= 0:
                    args['object_mask_id'] = mask_id_slot['value']

                data['args'] = json.dumps(args)

                # Execute action
                res = requests.post(url, data=data)
                res.raise_for_status()

            elif sys_dialogue_act == "request_label":
                # Agent querys cv engine and passes all masks to photoshop for display
                slots = sys_act['slots']

                # Get masks from slots
                masks = []
                for mask_slot in slots:
                    tup = (mask_slot['slot'],  mask_slot['value'])
                    masks.append(tup)

                data = {}
                data['control_type'] = "load_masks"
                args = {}
                args['masks'] = masks
                data['args'] = json.dumps(args)

                url = self.control_url

                res = requests.post(url, data=data)
                res.raise_for_status()

        # Retrieve Image
        res = requests.post(self.check_url, data={'intent': 'check'})
        res.raise_for_status()
        obj = res.json()

        backend_act.update(obj)

        self.observation = {}
        return backend_act
