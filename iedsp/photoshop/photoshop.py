import json
import logging
import requests
from urllib.parse import urljoin

from .sps import SimplePhotoshop

from ..core import SystemAct, PhotoshopAct
from ..util import find_slot_with_key, img_to_b64, build_slot_dict


logger = logging.getLogger(__name__)


def PhotoshopPortal(photoshop_config):
    use_sps = bool(int(photoshop_config['USE_SPS']))
    use_sps_client = bool(int(photoshop_config['USE_SPS_CLIENT']))
    sps_uri = photoshop_config['SPS_URI']

    if use_sps:
        if use_sps_client:
            return SimplePhotoshopClient(sps_uri)
        else:
            return SimplePhotoshopAgent()
    return None


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

    def observe(self, observation):
        """
        Observes system act
        """
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
                    if slot['slot'] == "mask_str" and slot.get('value'):
                        mask_str_slots.append(slot)

                # Process mask_str_slot to SimplePhotoshop arg format
                mask_strs = []
                for idx, mask_slot in enumerate(mask_str_slots):
                    tup = (str(idx),  mask_slot['value'])
                    mask_strs.append(tup)

                args = {}
                args['mask_strs'] = mask_strs

                result, msg = self.control(PhotoshopAct.LOAD_MASK_STRS, args)
                logger.debug("{}: {}, {}".format(
                    PhotoshopAct.LOAD_MASK_STRS, result, msg))

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

                if action == "control":
                    result, msg = self.control(intent, args)
                else:
                    result, msg = self.execute(intent, args)

        # Get slots
        image = self.get_image(False)
        b64_img_str = img_to_b64(image)

        masked_image = self.get_image(True)
        masked_b64_img_str = img_to_b64(masked_image)

        mask_strs = []
        for mask_idx, mask in self.get_masks():
            mask_str = img_to_b64(mask)
            mask_strs.append((mask_idx, mask_str))

        # Build return object
        photoshop_act = {}
        ps_act = {
            'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
            'slots': [
                build_slot_dict('b64_img_str', b64_img_str, 1.0),
                build_slot_dict('masked_b64_img_str', masked_b64_img_str, 1.0),
                build_slot_dict('mask_strs', mask_strs, 1.0)
            ]
        }

        photoshop_act = {}
        photoshop_act['photoshop_acts'] = [ps_act]
        self.observation = {}
        return photoshop_act


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
                    if slot['slot'] == "mask_str" and slot.get('value'):
                        mask_str_slots.append(slot)

                # Process mask_str_slot to SimplePhotoshop arg format
                mask_strs = []
                for idx, mask_slot in enumerate(mask_str_slots):
                    tup = (str(idx),  mask_slot['value'])
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
        data = {
            'action': 'control',
            'intent': 'check',
            'args': json.dumps({})
        }
        res = requests.post(self.action_uri, data=data)
        res.raise_for_status()
        check_obj = res.json()

        b64_img_str = check_obj.get("b64_img_str", "")
        masked_b64_img_str = check_obj.get("masked_b64_img_str", "")

        # Build return object
        photoshop_act = {}
        ps_act = {
            'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
            'slots': [
                build_slot_dict('b64_img_str', b64_img_str, 1.0),
                build_slot_dict('masked_b64_img_str', masked_b64_img_str, 1.0)
            ]
        }

        photoshop_act = {}
        photoshop_act['photoshop_acts'] = [ps_act]
        self.observation = {}
        return photoshop_act
