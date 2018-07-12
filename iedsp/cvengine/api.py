"""
    https://git.corp.adobe.com/mling/vision-engine
"""
import requests

from ..util import img_to_b64, b64_to_img

SELECT_URL = "http://isupreme:5100/selection"


class CVEngineAPI:
    def select(noun, b64_img_str):
        """ Calls Server and returns mask array
        """
        # Post request and get results
        data = {
            'imgstr': b64_img_str,
            'text': noun
        }
        response = requests.post(SELECT_URL, data=data)
        masks = response.json()
        return masks

    def fake_select(noun, b64_img_str):
        """Fake select 
        """
        return ["mask0_str", "mask1_str"]
