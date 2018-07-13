import json
import requests

from ..ontology import Ontology
from ..util import find_slot_with_key


class SimplePhotoshopAPI(object):
    def __init__(self, api_url="http://localhost:2005"):
        """Stores the api for connection
        """
        self.edit_url = api_url + "/edit"
        self.control_url = api_url + "/control"

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
        system_acts = self.observation.get('system_acts', list())
        for sys_act in system_acts:
            sys_dialogue_act = sys_act['dialogue_act']
            if sys_dialogue_act == "execute":  # Else, ignore

                slots = sys_act['slots']

                # Edit or Control
                _, action_slot = find_slot_with_key('action_type', slots)
                action_type = action_slot['value']
                if action_type in ["open", "load", "select", "redo", "undo"]:
                    action_type_key = 'control_type'
                    url = self.control_url
                elif action_type in ["adjust"]:
                    action_type_key = 'edit_type'
                    url = self.edit_url
                else:
                    raise ValueError(
                        "Unknown action_type: {}".format(action_type))

                slot_names = Ontology.get(action_type)

                # Build
                data = {}
                data[action_type_key] = action_type
                args = {}
                for slot_name in slot_names:
                    _, slot = find_slot_with_key(slot_name, slots)
                    args[slot_name] = slot['value']
                data['args'] = json.dumps(args)

                # Post to backend
                res = requests.post(url, data=data)
                obj = res.json()

        backend_act = {}
        backend_act['b64_img_str'] = obj['b64_img_str']
        return backend_act
