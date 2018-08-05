import logging

import requests
import urllib.parse

from ..util import img_to_b64, b64_to_img


class CVEngineClient(object):
    """
    Client to Computer Vision Engine  
    Github here: https://git.corp.adobe.com/mling/vision-engine
    Attributes:
        uri (str): cv engine uri
    """

    def __init__(self, cvengine_uri):
        self.uri = cvengine_uri

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
        select_uri = urllib.parse.urljoin(self.uri, 'selection')

        # Post request and get results
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
            response = requests.post(select_uri, data=data)
            response.raise_for_status()
            mask_strs = response.json()
        except Exception as e:
            print(e)
            mask_strs = []

        # Convert mask_strs to slots and set slot to index
        return mask_strs
