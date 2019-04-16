import logging
import sys
import time

import requests
import urllib.parse

from ..util import load_from_pickle

logger = logging.getLogger(__name__)


def VisionEnginePortal(visionengine_config):
    visionengine_name = visionengine_config['visionengine']
    args = {
        'uri': visionengine_config.get("uri"),
        'db_path': visionengine_config.get('database_path')
    }
    return builder(visionengine_name)(**args)


class BaseVisionEngine(object):
    """
    Base class for vision engine clients
    Defines methods that needs to be overridden
    Attributes:
        self.visionengine_uri (str): uri that needs to be queried
    """

    def __init__(self):
        raise NotImplementedError

    def select_object(self, b64_img_str, object, position=None, adjective=None, color=None):
        """ 
        Args:
            b64_img_str (str)
            object (str)
            position (str)
            adjective (str)
            color (str)
        Returns:
            mask_strs (list) : list of b64_img_strs
        """
        raise NotImplementedError


class DummyClient(BaseVisionEngine):
    """
    A dummy vision engine client for testing purposes
    """

    def select_object(self, b64_img_str, object, position=None, adjective=None, color=None):
        return []


class MingYangClient(BaseVisionEngine):
    """
    Client to MingYang's vision engine
    Github here: https://git.corp.adobe.com/mling/vision-engine
    """

    def __init__(self, **kwargs):
        self.uri = kwargs['uri']

    def select_object(self, b64_img_str=None, object=None, position=None, adjective=None, color=None):

        select_uri = urllib.parse.urljoin(self.uri, 'selection')

        # Build POST data according to MingYang's demo http://isupreme:5100/
        if position is None and adjective is None and color is None:
            data = {
                'imgstr': b64_img_str,
                'text': object
            }
        else:
            data = {
                'imgstr': b64_img_str,
                'noun': object,
            }
            if position is not None:
                data['position'] = position
            if adjective is not None:
                data['adjs'] = adjective
            if color is not None:
                data['color'] = color

        try:
            logger.info("Querying MingYang's CV engine")
            response = requests.post(select_uri, data=data)
            response.raise_for_status()
            mask_strs = response.json()
        except Exception as e:
            print(e)
            logger.info(e)
            mask_strs = []

        return mask_strs


class MaskRCNNClient(BaseVisionEngine):
    """
    Client to self-hosted Mask-RCNN server
    https://git.corp.adobe.com/tzlin/Mask_RCNN

    Supports object class detection
    """

    def __init__(self, **kwargs):
        self.uri = kwargs['uri']

    def select_object(self, b64_img_str, object, **kwargs):
        start_time = time.time()

        select_uri = urllib.parse.urljoin(self.uri, 'selection')

        # Unlike MingYan's engine, does not allow referring expressions
        data = {
            'imgstr': b64_img_str,
            'text': object
        }
        try:
            logger.info("Querying MaskRCNN server")
            response = requests.post(select_uri, data=data)
            response.raise_for_status()
            mask_strs = response.json()
        except Exception as e:
            print(e)
            logger.info(e)
            mask_strs = []

        end_time = time.time()

        print("Time taken: {} seconds".format(end_time-start_time))
        return mask_strs


class MAttNetClient(BaseVisionEngine):
    """
    Client to self-hosted MAttNet server

    Supports referring expression
    """

    def __init__(self, **kwargs):
        self.uri = kwargs['uri']

    def select_object(self, b64_img_str, object, **kwargs):
        """
        Args:
            b64_img_str: numpy image in base64 string format
            object: referring expression

        Returns:
            mask_strs: list of b64_mask_strs
        """

        # Unlike MingYan's engine, does not allow referring expressions
        data = {
            "expr": object,
            "b64_img_str": b64_img_str
        }
        try:
            logger.info("Querying MAttNet server")
            response = requests.post(self.uri, data=data)
            response.raise_for_status()
            mask_img_str = response.json()['mask_img_str']
            mask_strs = [mask_img_str]
        except Exception as e:
            print(e)
            logger.info(e)
            mask_strs = []
        return mask_strs


class VisionEngineDatabase(BaseVisionEngine):
    """
    Inferenced results from VisionEngine

    Attributes:
        db (dict): db is a dict of dicts
    """

    def __init__(self, **kwargs):
        self.db = load_from_pickle(kwargs['db_path'])

    def select_object(self, b64_img_str=None, object=None, position=None, adjective=None, color=None, **kwargs):
        if b64_img_str is None:
            #print("[visionengine] missing b64_img_str")
            logging.debug("[visionengine] missing b64_img_str")
            return []
        elif object is None:
            #print("[visionengine] missing object")
            logging.debug("[visionengine] missing object")
            return []
        elif b64_img_str not in self.db:
            #print("[visionengine] b64_img_str not in db")
            logger.debug("{} not in db".format(b64_img_str))
            return []
        mask_strs = self.db.get(b64_img_str).get(object)
        return mask_strs


def builder(string):
    """
    Gets visionengine class with string
    """
    return getattr(sys.modules[__name__], string)
