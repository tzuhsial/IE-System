"""
    https://git.corp.adobe.com/mling/vision-engine
"""
import requests

from ..util import img_to_b64, b64_to_img

SELECT_URL = "http://isupreme:5100/selection"


class CVEngineAPI:
    def select_object(noun, b64_img_str):
        """ Calls Server and returns mask array
        """
        # Post request and get results
        data = {
            'imgstr': b64_img_str,
            'text': noun
        }
        try:
            response = requests.post(SELECT_URL, data=data)
            response.raise_for_status()
            masks = response.json()
        except Exception as e:
            print(e)
            masks = []
        return masks

    def fake_select_object(noun, b64_img_str):
        """Fake select 
        """
        return ["mask0_str", "mask1_str"]
