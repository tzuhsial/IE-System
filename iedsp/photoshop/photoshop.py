import json
import requests
from urllib.parse import urljoin


from .sps import SimplePhotoshop

from ..core import SystemAct
from ..ontology import ImageEditOntology
from ..util import find_slot_with_key, img_to_b64


def PhotoshopGateway(config):
    """
    Photoshop Getter
    """
    use_sps = bool(int(config['USE_SPS']))
    use_sps_client = bool(int(config['USE_SPS_CLIENT']))

    if use_sps:
        if use_sps_client:
            return SimplePhotoshopClient(config)
        else:
            return SimplePhotoshopAgent(config)
    else:
        raise NotImplementedError


def slots_to_args(slots):
    """
    Converts list of dicts to argument dict 
    """
    args = {}
    for slot_dict in slots:
        args[slot_dict['slot']] = slot_dict['value']
    return args


class SimplePhotoshopAgent(SimplePhotoshop):
    """
       This class inherits SimplePhotoshop and adds agent methods
       For more methods, please look into ./sps directory 
       or SimplePhotoshop repo
       Since this is SimplePhotoshop, it uses the ImageEditOntology
    """

    def __init__(self, config):
        """
        Calls parent class for initialization
        """
        super(SimplePhotoshopAgent, self).__init__()

    def reset(self):
        """
        Reset
        """
        self.observation = {}

    def observe(self, observation):
        """
        Observes system act
        """
        self.observation = observation

    def act(self):
        """ 
        Act according to observation from system
        Returns:
            photoshop_act (dict):
        """
        system_acts = self.observation.get('system_acts', list())
        for sys_act in system_acts:
            if sys_act['dialogue_act'] == SystemAct.EXECUTE:
                # Execute command
                intent = sys_act['intent']
                slots = sys_act['slots']
                assert intent in ImageEditOntology.domain_type_map

                intent_type = ImageEditOntology.domain_type_map.get(intent)
                args = slots_to_args(slots)
                if intent_type == "control":
                    result = self.control(intent, args)
                elif intent_type == "edit":
                    result = self.execute(intent, args)
                else:
                    raise ValueError(
                        "Unknown intent_type: {}".format(intent_type))

            elif sys_act['dialogue_act'] == SystemAct.REQUEST_LABEL:
                # System provide masks to the Photoshop for the user to label
                if sys_act.get("slots") is not None:
                    intent = "load_masks"
                    args = slots_to_args(sys_act['slots'])
                    self.control(intent, args)

        # Build return object
        photoshop_act = {}
        if self.getImage() is not None:
            b64_img_str = img_to_b64(self.getImage())
            photoshop_act['b64_img_str'] = b64_img_str
        return photoshop_act


class SimplePhotoshopClient(object):
    """
        API interface to Photoshop
    """

    def __init__(self, config):
        """Stores the api for connection
        """
        # Configuration
        self.photoshop_uri = config["PHOTOSHOP_URI"]

        self.edit_url = urljoin(self.photoshop_uri, "edit")
        self.control_url = urljoin(self.photoshop_uri, "/control")
        self.check_url = urljoin(self.photoshop_uri, "/check")

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

                intent_type = sys_act['intent']

                if intent_type in ["open", "load", "select_object", "select_object_mask_id", "redo", "undo"]:
                    intent_type_key = 'control_type'
                    url = self.control_url
                elif intent_type in ["adjust"]:
                    intent_type_key = 'edit_type'
                    url = self.edit_url
                else:
                    raise ValueError(
                        "Unknown intent_type: {}".format(intent_type))

                slot_names = Ontology.getArgumentsWithIntent(intent_type)

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
            else:
                assert s

        # Retrieve Image
        res = requests.post(self.check_url, data={'intent': 'check'})
        res.raise_for_status()
        obj = res.json()

        backend_act.update(obj)

        self.observation = {}
        return backend_act
