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

    def select_object(self, noun, b64_img_str):
        """ 
        Calls Server and returns mask array
        """
        select_uri = urllib.parse.urljoin(self.uri, 'selection')

        # Post request and get results
        data = {
            'imgstr': b64_img_str,
            'text': noun
        }
        try:
            response = requests.post(select_uri, data=data)
            response.raise_for_status()
            mask_strs = response.json()
        except Exception as e:
            print(e)
            mask_strs = []

        # Convert mask_strs to slots and set slot to index
        mask_str_slots = []
        for idx, mask_str in enumerate(mask_strs):
            mask_str_slot = {'slot': idx, 'value': mask_str}
            mask_str_slots.append(mask_str_slot)
        return mask_str_slots


if __name__ == "__main__":
    pass
