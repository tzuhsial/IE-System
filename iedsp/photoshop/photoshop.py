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

    @property
    def name(self):
        return self.__class__.__name__

    def reset(self):
        pass

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
                _, action_slot = find_slot_with_key('action_type', slots)
                action_type = action_slot['value']
                if action_type in ["open", "load", "select_object", "select_object_mask_id", "redo", "undo"]:
                    action_type_key = 'control_type'
                    url = self.control_url
                elif action_type in ["adjust"]:
                    action_type_key = 'edit_type'
                    url = self.edit_url
                else:
                    raise ValueError(
                        "Unknown action_type: {}".format(action_type))

                slot_names = Ontology.get(action_type)

                # Build post data to backend
                data = {}
                data[action_type_key] = action_type
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
                # TODO: handle error upon request
                res = requests.post(url, data=data)
                res.raise_for_status()
            elif sys_dialogue_act == "request_label":
                # Agent will need to pass all the masks to photoshop
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
        res = requests.post(self.check_url, data={'action_type': 'check'})
        res.raise_for_status()
        obj = res.json()

        backend_act.update(obj)
        return backend_act
