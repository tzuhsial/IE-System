import logging
import sys

import requests
import urllib.parse

logger = logging.getLogger(__name__)


def VisionEnginePortal(visionengine_config):
    client = visionengine_config['VISIONENGINE_CLIENT']
    uri = visionengine_config["VISIONENGINE_URI"]
    return builder(client)(uri)


class BaseVisionEngineClient(object):
    """
    Base class for vision engine clients
    Defines methods that needs to be overridden
    Attributes:
        self.visionengine_uri (str): uri that needs to be queried
    """

    def __init__(self, uri):
        self.uri = uri

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


class DummyClient(BaseVisionEngineClient):
    """
    A dummy vision engine client for testing purposes
    """

    def select_object(self, b64_img_str, object, position=None, adjective=None, color=None):
        return []


class MingYangClient(BaseVisionEngineClient):
    """
    Client to MingYang's vision engine
    Github here: https://git.corp.adobe.com/mling/vision-engine
    """

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


class MaskRCNNClient(BaseVisionEngineClient):
    """
    Client to self-hosted Mask-RCNN server
    https://git.corp.adobe.com/tzlin/Mask_RCNN

    Supports object class detection
    """

    def select_object(self, b64_img_str, object, **kwargs):

        select_uri = urllib.parse.urljoin(self.visionengine_uri, 'selection')

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

        return mask_strs


def builder(string):
    """
    Gets visionengine class with string
    """
    return getattr(sys.modules[__name__], string)
