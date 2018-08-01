from urllib.parse import urlparse


def is_uri(uri):
    """https://stackoverflow.com/questions/7160737/python-how-to-validate-a-url-in-python-malformed-or-not
    """
    try:
        result = urlparse(uri)
        return result.scheme and result.netloc and result.path
    except:
        return False


def is_probability(number):
    try:
        number = float(number)
        return number >= 0 and number <= 1
    except ValueError:
        return False


def is_str(object):
    return isinstance(object, str)


def is_int(object):
    return isinstance(object, int)
