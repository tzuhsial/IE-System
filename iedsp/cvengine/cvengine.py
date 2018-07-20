"""
    https://git.corp.adobe.com/mling/vision-engine
"""
import requests
import urllib.parse

from ..util import img_to_b64, b64_to_img

SELECT_URL = "http://isupreme:5100/selection"


class CVEngineAPI(object):
    """
        API Interface to CV Engine
    """

    def __init__(self, cvengine_uri):
        self.uri = cvengine_uri

    def check_cvengine(self):
        return requests.head(self.uri).status_code == 200

    def select_object(self, noun, b64_img_str):
        """ Calls Server and returns mask array
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
            masks = response.json()
        except Exception as e:
            print(e)
            masks = []
        return masks

    def fake_select_object(self, noun, b64_img_str):
        """Fake select 
        """
        return ["mask0_str", "mask1_str"]


if __name__ == "__main__":
    pass
