import logging
import sys
import time

import requests
import urllib.parse

from ..core import UserAct
from .. import util

logger = logging.getLogger(__name__)


def VisionEnginePortal(visionengine_config):
    visionengine_name = visionengine_config['visionengine']
    args = {
        'uri': visionengine_config.get("uri"),
        'db_path': visionengine_config.get('database_path')
    }
    return builder(visionengine_name)(**args)


class MAttNetClient(object):
    """
    Client to self-hosted MAttNet server
    """

    def __init__(self, **kwargs):
        self.uri = kwargs['uri']

    def reset(self):
        pass

    def observe(self, observation):
        self.observation = observation

    def act(self):

        system_acts = self.observation.get('system_acts', [])
        query_act = system_acts[0]

        query_slots = query_act['slots']

        object_slot = util.find_slot_with_key('object', query_slots)
        refer_expr = object_slot['value']

        img_str_slot = util.find_slot_with_key(
            'original_b64_img_str', query_slots)
        b64_img_str = img_str_slot['value']

        mask_strs = self.select_object(b64_img_str, refer_expr)

        assert len(mask_strs) <= 1, "MattNet returns at most 1 result!"

        # Handles no mask_str case
        slots = [util.build_slot_dict(
            'object_mask_str', mask_str, 0.5) for mask_str in mask_strs]

        utterance = "Detected {} result.".format(len(mask_strs))

        engine_act = {
            'dialogue_act': util.build_slot_dict('dialogue_act', UserAct.INFORM, 1.0),
            'slots': slots
        }

        vis_act = {
            'visionengine_acts': [engine_act],
            'visionengine_utterance': utterance,
            'dialogue_act': util.build_slot_dict('dialogue_act', UserAct.INFORM, 1.0),
            'slots': slots,
        }

        return vis_act

    def select_object(self, b64_img_str, refer_expr):
        """
        Args:
            b64_img_str: numpy image in base64 string format
            refer_expr: referring expression

        Returns:
            mask_strs: list of b64_mask_strs
        """
        data = {
            "expr": refer_expr,
            "b64_img_str": b64_img_str
        }
        try:
            logger.info("Querying MAttNet server")
            response = requests.post(self.uri, data=data)
            response.raise_for_status()
            mask_img_str = response.json()['mask_img_str']
            mask_strs = [mask_img_str]
        except Exception as e:
            print("Query MattNet server failed!")
            logger.info(e)
            mask_strs = []
        return mask_strs


def builder(string):
    """
    Gets visionengine class with string
    """
    return getattr(sys.modules[__name__], string)
