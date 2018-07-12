"""

    https://git.corp.adobe.com/mling/vision-engine
"""
import requests

from ..utils import img_to_b64, b64_to_img

SELECT_URL = "http://isupreme:5100/selection"

def select(img, noun):
    """ Calls Server and returns mask array
    """
    # Convert to base64 str
    b64_img_str = img_to_b64(img)

    # Post request and get results
    data = {
        'imgstr': b64_img_str,
        'text': noun
    }
    response = requests.post(SELECT_URL, data=data)
    results = response.json()

    # Get mask array
    masks = []
    for mask_str in results:
        mask = b64_to_img(mask_str)
        masks.append(mask)

    return masks
