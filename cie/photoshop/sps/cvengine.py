from urllib.parse import urljoin

import requests

from .utils import img_to_b64, b64_to_img


class CVEngineClient(object):
    def __init__(self, uri="http://isupreme:5100/"):
        self.uri = uri
        # Endpoints
        self.selection_uri = urljoin(uri, 'selection')

    def select(self, img, noun):
        """ 
        Calls Server and returns mask array
        Returns:
            masks (list): list of mask strings
        """
        print('noun', noun)
        # Convert to base64 str
        b64_img_str = img_to_b64(img)

        # Post request and get results
        data = {
            'imgstr': b64_img_str,
            'text': noun
        }

        try:
            response = requests.post(self.selection_uri, data=data)
            response.raise_for_status()
            results = response.json()
        except Exception as e:
            print(e)
            results = []

        print('selection:', len(results))
        # Get mask array
        masks = []
        for mask_str in results:
            mask = b64_to_img(mask_str)
            masks.append(mask)

        return masks
