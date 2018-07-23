import configparser
from urllib.parse import urlparse


def uri_validator(uri):
    """https://stackoverflow.com/questions/7160737/python-how-to-validate-a-url-in-python-malformed-or-not
    """
    try:
        result = urlparse(uri)
        return result.scheme and result.netloc and result.path
    except:
        return False


def probability_validator(number):
    try:
        number = float(number)
        return number >= 0 and number <= 1
    except ValueError:
        return False


def test_config_format():
    """ Test config format
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Backend
    assert uri_validator(config['BACKEND']['CVENGINE_URI'])
    assert uri_validator(config['BACKEND']['PHOTOSHOP_URI'])

    # Error channel
    assert probability_validator(config['CHANNEL']['CONF_MEAN'])
    assert probability_validator(config['CHANNEL']['CONF_STD'])

    # System
    assert probability_validator(config['SYSTEM']['CONFIRM_THRESHOLD'])

    # User
    assert probability_validator(config['USER']['SELECT_OBJECT_FIRST'])
