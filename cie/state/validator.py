import base64
import logging
import os
import sys
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseValidator(object):
    pass


class URIValidator(BaseValidator):
    def __call__(self, obj):
        try:
            result = urlparse(obj)
            return result.scheme and result.netloc and result.path
        except:
            return False


class ProbabilityValidator(BaseValidator):
    def __call__(self, obj):
        try:
            obj = float(obj)
            return 0. <= obj <= 1.
        except ValueError:
            return False


class StringValidator(BaseValidator):
    def __call__(self, obj):
        return isinstance(obj, str)


class IntegerValidator(BaseValidator):
    def __call__(self, obj):
        return isinstance(obj, int)


class PathValidator(BaseValidator):
    def __call__(self, obj):
        if not isinstance(obj, str):
            return False
        return os.path.exists(obj)


class B64ImgStrValidator(BaseValidator):
    def __call__(self, obj):
        try:
            buf = base64.b64decode(obj)
            nparr = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            return False
        return True


class BooleanValidator(BaseValidator):
    def __call__(self, obj):
        return isinstance(obj, bool)


def builder(string):
    # Returns Validator Objects
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError as e:
        print(e)
        logger.error("Unknown validator: {}".format(string))
        return None
